import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
    static_image_mode = True,
    model_complexity = 2
) as holistic:
    image = cv2.imread("IMG_8742.jpeg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = holistic.process(image_rgb)
    
    # ROSTRO
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2))
    
    # MANO IZQ (azul)
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
    
    # MANO DER (verde)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(57, 143, 0), thickness=2))
    
    # POSE
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

    
    
    cv2.imshow("Image", image)

    
    # PLOT : PUNTOS DE REFERENCIA Y CONEXIONES EN MATPLOTLIB 3D
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS
    )    

    cv2.waitKey(0)
cv2.destroyAllWindows()
    
    