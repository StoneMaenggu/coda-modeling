import cv2
import mediapipe as mp
import numpy as np

class I2P_Module:
    def __init__(self):
        # Initialize MediaPipe Pose and Hands
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.use_pose_idx = [12,14,16,11,13,15]
        self.use_hand_idx = [0,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20]
        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils

    def predict(self, image, vis=False):
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to extract pose landmarks
        pose_results = self.pose.process(image_rgb)

        # Process the image to extract hand landmarks
        hand_results = self.hands.process(image_rgb)
        if vis:
            # Draw pose landmarks on the image
            annotated_image = image.copy()
            
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Overlay indices for pose landmarks
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    x = int(landmark.x * annotated_image.shape[1])
                    y = int(landmark.y * annotated_image.shape[0])
                    cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            
            # Draw hand landmarks on the image
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Overlay indices for hand landmarks
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x = int(landmark.x * annotated_image.shape[1])
                        y = int(landmark.y * annotated_image.shape[0])
                        cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # Save the annotated image
            cv2.imwrite('sample.jpg', annotated_image)

        # Extract pose landmarks
        pose_landmarks = []
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                pose_landmarks.append((landmark.x, landmark.y))

        # Extract hand landmarks
        left_hand_landmarks = []
        right_hand_landmarks = []
        if hand_results.multi_handedness:
            for i, hand_handedness in enumerate(hand_results.multi_handedness):
                hand_landmarks = hand_results.multi_hand_landmarks[i]
                handedness = hand_handedness.classification[0].label
                hand_landmark_list = []
                for landmark in hand_landmarks.landmark:
                    hand_landmark_list.append((landmark.x, landmark.y))
                
                if handedness == 'Left':
                    left_hand_landmarks = hand_landmark_list
                else:
                    right_hand_landmarks = hand_landmark_list

        ret = []
        if len(pose_landmarks):
            ret.append(np.array(pose_landmarks)[self.use_pose_idx,:])
        else:
            ret.append(np.zeros((len(self.use_pose_idx),2)))
        if len(left_hand_landmarks):
            ret.append(np.array(left_hand_landmarks)[self.use_hand_idx,:])
        else:
            ret.append(np.zeros((len(self.use_hand_idx),2)))
        if len(right_hand_landmarks):
            ret.append(np.array(right_hand_landmarks)[self.use_hand_idx,:])
        else:
            ret.append(np.zeros((len(self.use_hand_idx),2)))
        
        return np.concatenate(ret,0)

    

if __name__ == '__main__':
    i2p_module = I2P_Module()

    # Example usage
    image_path = '/home/horang1804/Downloads/sample.jpg'  # Replace with your image path
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")


    pose = i2p_module.predict(image, vis=False)
