import os
import cv2
import av
import pandas as pd
import argparse

def visualize(video_path, csv_path, grid_size=7):
    print(f"Loading results from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"[ERROR] Result file {csv_path} not found. Run the tracker first.")
        return

    # Load tracking data
    df = pd.read_csv(csv_path)
    # Filter only frames where we actually detected or propagated a box
    df = df[df['confidence'] > 0]
    if df.empty:
        print("[WARN] No detections with confidence > 0 were found in the CSV.")
        print("       Check whether the tracker ran with a valid model and produced real detections.")
        return
    
    # Group by PTS for easy lookup
    boxes_by_pts = df.groupby('pts')

    print(f"Opening video {video_path} for full reconstruction...")
    container = av.open(video_path)
    
    # Prepare output folder
    output_dir = os.path.join(os.path.dirname(video_path), "visualization_frames")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare video writer for MP4 video output
    stream = container.streams.video[0]
    w_img, h_img = stream.width, stream.height
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    
    output_mp4 = r"visuals\tracking_visualization.mp4"
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_mp4, fourcc, fps, (w_img, h_img))

    print("Starting visualization loop...")
    for frame in container.decode(video=0):
        pts = frame.pts
        
        # Only visualize frames that are present in the I/P/B tracking results
        if pts not in boxes_by_pts.groups:
            continue
            
        f_type = frame.pict_type.name if hasattr(frame.pict_type, 'name') else str(frame.pict_type)
        
        # Get pixels (this works because we use a standard av container)
        img = frame.to_ndarray(format='bgr24')

        if pts in boxes_by_pts.groups:
            frame_boxes = boxes_by_pts.get_group(pts)
            
            for _, row in frame_boxes.iterrows():
                # Convert grid coordinates back to pixel coordinates
                # cx/cy are in range [0, grid_size]
                cx_rel = row['cx'] / grid_size
                cy_rel = row['cy'] / grid_size
                w_rel = row['w'] / grid_size
                h_rel = row['h'] / grid_size

                # Absolute pixel locations
                xmin = int((cx_rel - w_rel/2) * w_img)
                ymin = int((cy_rel - h_rel/2) * h_img)
                xmax = int((cx_rel + w_rel/2) * w_img)
                ymax = int((cy_rel + h_rel/2) * h_img)

                # Color: Green for Detection (I), Blue for Propagation (P/B)
                color = (0, 255, 0) if row['source'] == 'DET' else (255, 255, 0)
                
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                
                label = f"{row['frame_type']} ({row['confidence']:.2f})"
                cv2.putText(img, label, (xmin, max(ymin - 10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show frame type on screen
        cv2.putText(img, f"Frame: {pts} Type: {f_type}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Save image frame and write to MP4 video
        save_path = os.path.join(output_dir, f"render_{pts:06d}.jpg")
        cv2.imwrite(save_path, img)
        if writer.isOpened():
            writer.write(img)

    if writer:
        writer.release()

    cv2.destroyAllWindows()
    print(f"\nDone. Visualization frames saved in: {output_dir}")
    print(f"Visualization video output saved in: {output_mp4}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', required=True)
    parser.add_argument('--csv', '-c', default=r"tracking_outputs\tracking_results.csv")
    parser.add_argument('--grid', '-g', type=int, default=7)
    args = parser.parse_args()

    visualize(args.video, args.csv, args.grid)
