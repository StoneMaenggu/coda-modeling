import cv2
from image2pose.i2p_module import I2P_Module
from pose2gloss.p2g_module import P2G_Module
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
def create_db(i2p,
              p2g,
              base_path, 
              save_path,
              n_feat,
              only_vis=True):
    gloss_feat_db = pd.DataFrame(columns=['pose_path','gloss']+[i for i in range(n_feat)])
    if base_path is not None:
        video_list = glob.glob(os.path.join(base_path,'*'))
        for db_idx, video_path in enumerate(tqdm(video_list)):
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            gloss = os.path.split(video_path)[1].split('_')[0]
            # Check if the video file was successfully opened
            if not cap.isOpened():
                print("Error: Could not open video file.")
                exit()
            # else:
            #     pose_list = glob.glob(os.path.join('./dataset/pose','*'))
            #     for db_idx, pose_path in enumerate(tqdm(pose_list)):
            #         # Open the video file
            #         cap = cv2.VideoCapture(pose_path)
            #         gloss = os.path.split(pose_path)[1].split('_')[0]
            #         # Check if the video file was successfully opened
            #         if not cap.isOpened():
            #             print("Error: Could not open video file.")
            #             exit()


            # Read and display video frames
            seq_image = []
            seq_pose = []
            while True:
                ret, frame = cap.read()  # Read a frame from the video
                if not ret:
                    break  # Exit the loop if no more frames are available
                seq_image.append(frame)

                pose = i2p.predict(frame)

                seq_pose.append(pose)

            seq_pose = np.stack(seq_pose)
            seq_len, n_feat, n_dim = seq_pose.shape
            start_idx = seq_len//2-45//2
            input_pose = seq_pose[start_idx:start_idx+45]

            feat = p2g.get_features(input_pose).squeeze()
            file_name = os.path.split(video_path)[-1].split('.')[0]
            if save_path is not None:
                np.save(os.path.join(save_path,file_name+'.npy'),input_pose)

            if only_vis:
                output_directory = './samples'
                for idx, (img, pose) in enumerate(zip(seq_image, seq_pose)):
                    # Ensure the image has the expected shape
                    img = np.array(img)  # Convert to a numpy array if it's not already
                    h,w,c = img.shape
                    if img.shape != (1080, 1920, 3):
                        print(f"Skipping image {idx}: shape {img.shape} does not match expected (1080, 1620, 3)")
                        continue
                    
                    # Draw the pose on the image
                    for point_id, point in enumerate(pose):
                        x, y = int(point[0]*w), int(point[1]*h)
                        if 0 <= x < 1920 and 0 <= y < 1080:  # Ensure the point is within image bounds
                            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle
                            # Put the index number next to the pose point
                            cv2.putText(img, str(point_id), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2, cv2.LINE_AA)

                    # Save the image
                    output_path = os.path.join(output_directory, f"output_{idx:04d}.jpg")
                    cv2.imwrite(output_path, img)
                    print(f"Saved {output_path}")
                break
            pose_dir=os.path.join('/home/horang1804/Dolmaggu/coda-modeling/Inference/dataset/pose', file_name+'.npy')
            gloss_feat_db.loc[db_idx] = (pose_dir,gloss,*feat.numpy())
    gloss_feat_db.to_csv('./gloss_db.csv',index=False)

    print('end')


if __name__ == '__main__':
    i2p = I2P_Module()
    p2g = P2G_Module('/home/horang1804/Dolmaggu/coda-modeling/Inference/pose2gloss/checkpoints/epoch_400.pt',None)
    base_path = './dataset/video'
    save_path = './dataset/pose'
    # Path to the video file
    
    create_db(i2p,
              p2g,
              base_path, 
              save_path,
              32,
              only_vis=False)