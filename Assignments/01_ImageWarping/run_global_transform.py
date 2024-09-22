import gradio as gr
import cv2
import numpy as np
import math

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    # rotation_1=rotation/360*2*math.pi
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         x=i-image.shape[0]//2
    #         y=j-image.shape[1]//2
    #         x_new=image.shape[0]//2+int(math.cos(rotation_1)*x+math.sin(rotation_1)*y)
    #         y_new=int(image.shape[1]//2+-math.sin(rotation_1)*x+math.cos(rotation_1)*y)
    #         if 0<=x_new<image.shape[0] and 0<=y_new<image.shape[1] :
    #             transformed_image[i,j]=image[x_new,y_new]
    rows,cols = image.shape[:2]
    M1 = to_3x3(cv2.getRotationMatrix2D((rows//2,cols//2),rotation,scale))
    M2 = to_3x3(np.float32([[1,0,translation_x],[0,1,translation_y]]))
    M=(M2 @M1)[:2,:]
    transformed_image = cv2.warpAffine(image,M[:2,:],(cols,rows))
    if flip_horizontal:
        transformed_image=transformed_image[:, ::-1]
    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
