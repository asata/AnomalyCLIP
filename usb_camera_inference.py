import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from PIL import Image
import time
import cv2
import os
import random
import numpy as np
from utils import get_transform, normalize
from scipy.ndimage import gaussian_filter
from datetime import datetime
import gc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def save_visualized_result(image, anomaly_map, img_size, save_path, prefix=""):
    # RGB image expected
    vis = cv2.resize(image, (img_size, img_size))
    mask = normalize(anomaly_map[0])
    vis_result = apply_ad_scoremap(vis, mask)
    
    # Save original and result
    orig_save = os.path.join(save_path, f"{prefix}_orig.png")
    res_save = os.path.join(save_path, f"{prefix}_result.png")
    
    cv2.imwrite(orig_save, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(res_save, cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR))
    return res_save

def run_realtime_inference(args):
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- [INFO] Using device: {device} ---")

    # 1. Model Preparation (Only Once)
    print("--- [INFO] Loading Model... (Stay in memory) ---")
    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx, 
        "learnabel_text_embedding_depth": args.depth, 
        "learnabel_text_embedding_length": args.t_n_ctx
    }
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    # Prepare Global Prompts
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    preprocess, _ = get_transform(args)
    print("--- [INFO] Model ready for real-time inference ---")

    # 2. Camera Setup
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print(f"--- [ERROR] Cannot open camera index {args.cam_id} ---")
        return

    # Set Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Create Save Directory
    os.makedirs(args.save_path, exist_ok=True)

    print(f"--- [INFO] Starting loop with interval: {args.interval}s (Press Ctrl+C to stop) ---")

    loop_count = 0
    try:
        while True:
            loop_count += 1
            loop_start = time.time()
            
            # Flush Buffer (Capture latest frame)
            for _ in range(5): cap.grab()
            ret, frame = cap.read()
            if not ret:
                print("--- [WARNING] Failed to capture image ---")
                time.sleep(1)
                continue

            # Capture Time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Step 1: Preprocessing
            start_proc = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            img_tensor = preprocess(pil_img)
            image = img_tensor.reshape(1, 3, args.image_size, args.image_size).to(device)
            if device == "cuda": torch.cuda.synchronize()
            proc_time = time.time() - start_proc

            # Step 2: Inference
            start_inf = time.time()
            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                text_probs = image_features @ text_features.permute(0, 2, 1)
                text_probs = (text_probs/0.07).softmax(-1)
                anomaly_score = text_probs[0, 0, 1].item() # Anomaly probability
                
                anomaly_map_list = []
                for idx, patch_feature in enumerate(patch_features):
                    if idx >= args.feature_map_layer[0]:
                        patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                        similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                        anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                        anomaly_map_list.append(anomaly_map)

                anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
                anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0)
            
            if device == "cuda": torch.cuda.synchronize()
            inf_time = time.time() - start_inf

            # Step 3: Save & Logging
            start_save = time.time()
            res_path = save_visualized_result(rgb_frame, anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, prefix=timestamp)
            save_time = time.time() - start_save

            status = "ANOMALY" if anomaly_score > args.threshold else "NORMAL"
            
            print(f"[{timestamp}] Status: {status:7} | Score: {anomaly_score:.4f} | Inf: {inf_time:.3f}s | Save: {save_time:.3f}s | Saved: {res_path}")

            # Step 4: Periodic Memory Cleanup (Every 50 iterations)
            if loop_count % 50 == 0:
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                # print(f"--- [INFO] Periodic memory cleanup performed (Loop: {loop_count}) ---")

            # Wait for next interval
            elapsed = time.time() - loop_start
            wait_time = max(0.1, args.interval - elapsed)
            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\n--- [INFO] Stop signal received. Releasing camera... ---")
    finally:
        cap.release()
        print("--- [INFO] Camera released. Exit. ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP Real-time USB Camera Inference")
    # Camera settings
    parser.add_argument("--cam_id", type=int, default=0, help="camera index (/dev/videoX)")
    parser.add_argument("--interval", type=int, default=10, help="capture interval in seconds")
    parser.add_argument("--threshold", type=float, default=0.5, help="anomaly threshold")
    
    # Paths
    parser.add_argument("--save_path", type=str, default="./results/realtime", help="path to save results")
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/9_12_4_multiscale/epoch_15.pth', help='path to checkpoint')
    
    # Model parameters (Match with training/test)
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="prompt learner depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="prompt learner n_ctx")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="prompt learner t_n_ctx")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--sigma", type=int, default=4)

    args = parser.parse_args()
    run_realtime_inference(args)

