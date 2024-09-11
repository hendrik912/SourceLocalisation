import os
import mne
import utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gc

from mne import read_source_estimate
from mne.datasets import sample, eegbci, fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, write_inverse_operator, read_inverse_operator, apply_inverse_epochs, read_inverse_operator
from mne import read_source_estimate, SourceEstimate
from mne.minimum_norm import make_inverse_operator, write_inverse_operator, read_inverse_operator, apply_inverse_epochs, apply_inverse, read_inverse_operator
from PIL import Image, ImageDraw
from PIL import Image, ImageDraw, ImageFont
import joblib

# Ensure the Qt bindings are available and create a QApplication instance if not already created
try:
    from PyQt5.QtWidgets import QApplication
    if QApplication.instance() is None:
        app = QApplication([])
except ImportError:
    raise ImportError("Please install PyQt5 by running `pip install pyqt5`")


# Ensure the Qt bindings are available
mne.viz.set_3d_backend('pyvistaqt')

matplotlib.use('tkagg')

subjects_dir = r"C:\Users\hendr\mne_data\MNE-fsaverage-data" 

# --------------------------------------------------------------------------

def delete_images(image_paths):
    for image_path in image_paths:
        try:
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        except FileNotFoundError:
            print(f"File not found: {image_path}")
        except Exception as e:
            print(f"Error deleting {image_path}: {e}")

# --------------------------------------------------------------------------

def add_text_to_image(image_path, output_path, text, location, font_path=None, font_size=20, font_rotation=0):
    # Open the image
    img = Image.open(image_path)
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Load the font
    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)
    
    # Calculate text size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = (text_bbox[3] - text_bbox[1])*2
    
    # Create a new blank image for the rotated text
    text_img = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    
    # Add text to the blank image
    text_draw.text((0, 0), text, font=font, fill="black")
    
    # Rotate the text image
    rotated_text_img = text_img.rotate(font_rotation, resample=Image.Resampling.BICUBIC, expand=True)
    
    # Calculate the position to paste the rotated text
    text_x, text_y = location
    
    if font_rotation == 0:
        text_x -= text_width//2
    elif font_rotation == 90:
        text_y -= text_height//2
        
    # Paste the rotated text image onto the original image
    img.paste(rotated_text_img, (text_x, text_y), rotated_text_img)
    
    # Save the modified image
    img.save(output_path)

# --------------------------------------------------------------------------

def concatenate_images1(paths_X, paths_R, output_path, min_max_X=(0,1), min_max_R=(0,1), include_cb=True, offset=(100,100)):
    
    X_imgs = []
    R_imgs = []
    
    downscale_factor = 5
    
    for path_X, path_R in zip(paths_X, paths_R):
        X_imgs.append(Image.open(path_X))
        R_imgs.append(Image.open(path_R))
        
    total_width = sum([img.width for img in X_imgs]) + offset[0]
    
    if include_cb:
        
        width = 50 // downscale_factor
        output_path_cb_X = output_path[:-len(".png")] + "_X_cb.png"
        output_path_cb_R = output_path[:-len(".png")] + "_R_cb.png"
        
        create_colorbar_image(output_path_cb_X, colormap='coolwarm', orientation='vertical', width=width, height=X_imgs[0].height // downscale_factor, vmin=min_max_X[0], vmax=min_max_X[1])
        create_colorbar_image(output_path_cb_R, colormap='coolwarm', orientation='vertical', width=width, height=R_imgs[0].height // downscale_factor, vmin=min_max_R[0], vmax=min_max_R[1])
        
        img_cb_X = Image.open(output_path_cb_X)
        img_cb_R = Image.open(output_path_cb_R)

        total_width += width*downscale_factor*7
        
    total_height = X_imgs[0].height * 2 + offset[1]
    
    concatenated_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    x_pos = offset[0]
    
    for (X_img, R_img) in zip(X_imgs, R_imgs):
        
        concatenated_image.paste(X_img, (x_pos, offset[1]))
        concatenated_image.paste(R_img, (x_pos, X_img.height + offset[1]))
        x_pos += X_img.width

    new_width = concatenated_image.width // downscale_factor
    new_height = concatenated_image.height // downscale_factor
    
    concatenated_image = concatenated_image.resize((new_width, new_height),  Image.Resampling.LANCZOS)

    if include_cb: # colorbar
        concatenated_image.paste(img_cb_X, (concatenated_image.width-55, 5))
        concatenated_image.paste(img_cb_R, (concatenated_image.width-55, 5+concatenated_image.height//2))
        delete_images([output_path_cb_X, output_path_cb_R])

    concatenated_image.save(output_path)
    

def concatenate_images(paths_X, paths_R, output_path, min_max_X=(0, 1), min_max_R=(0, 1), include_cb=True, offset=(100, 100)):
    
    X_imgs = []
    R_imgs = []
    
    downscale_factor = 5
    
    for path_X, path_R in zip(paths_X, paths_R):
        X_imgs.append(Image.open(path_X))
        R_imgs.append(Image.open(path_R))
    
    max_width = max([img.width for img in X_imgs + R_imgs]) * 2 + offset[0]
    
    # if include_cb:
    #     width = 50 // downscale_factor
    #     output_path_cb_X = output_path[:-len(".png")] + "_X_cb.png"
    #     output_path_cb_R = output_path[:-len(".png")] + "_R_cb.png"
    #     create_colorbar_image(output_path_cb_X, colormap='coolwarm', orientation='vertical', width=width, height=X_imgs[0].height // downscale_factor, vmin=min_max_X[0], vmax=min_max_X[1])
    #     create_colorbar_image(output_path_cb_R, colormap='coolwarm', orientation='vertical', width=width, height=R_imgs[0].height // downscale_factor, vmin=min_max_R[0], vmax=min_max_R[1])
    #     img_cb_X = Image.open(output_path_cb_X)
    #     img_cb_R = Image.open(output_path_cb_R)
    #     max_width += width * downscale_factor * 7
        
    total_height = sum([img.height for img in X_imgs + R_imgs]) + offset[1]
    
    concatenated_image = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
    
    y_pos = offset[1]
    
    for X_img, R_img in zip(X_imgs, R_imgs):
        
        concatenated_image.paste(X_img, (offset[0], y_pos))
        
        concatenated_image.paste(R_img, (X_img.width + offset[0], y_pos))

        y_pos += X_img.height

    new_width = concatenated_image.width // downscale_factor
    new_height = concatenated_image.height // downscale_factor
    
    concatenated_image = concatenated_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # if include_cb:  # colorbar
    #     concatenated_image.paste(img_cb_X, (concatenated_image.width - 55, 5))
    #     concatenated_image.paste(img_cb_R, (concatenated_image.width - 55, 5 + concatenated_image.height // 2))
    #     delete_images([output_path_cb_X, output_path_cb_R])

    concatenated_image.save(output_path)

# --------------------------------------------------------------------------

def crop_image_pillow(image_path, output_path, left=0, right=0, top=0, bottom=0):

    with Image.open(image_path) as img:
        width, height = img.size

        left_crop = left
        right_crop = width - right
        top_crop = top
        bottom_crop = height - bottom

        if left_crop < 0:
            left_crop = 0
        if right_crop > width:
            right_crop = width
        if top_crop < 0:
            top_crop = 0
        if bottom_crop > height:
            bottom_crop = height

        img_cropped = img.crop((left_crop, top_crop, right_crop, bottom_crop))

        img_cropped.save(output_path)
        print(f"Cropped image saved as {output_path}")

# --------------------------------------------------------------------------

def plot_source_estimate(stc, subjects_dir, hemisphere="rh", time_viewer=True, data_min=None, data_max=None, output_file=None, view='lateral'):
    
    if data_min is None:
        data_min = stc.data.min()
    
    if data_max is None:
        data_max = stc.data.max()

    # Define plotting parameters with adjusted color limits
    surfer_kwargs = dict(
        hemi=hemisphere,
        subjects_dir=subjects_dir,
        clim=dict(kind="value", lims=[data_min, (data_min + data_max) / 2, data_max]),  # Adjusted based on data range
        views=view,
        time_viewer=time_viewer,
        size=(900, 900),
        smoothing_steps=10,
        colormap="coolwarm",
        title=method,
        background="white"
    )

    # Plot the entire source estimate with the time viewer enabled
    brain = stc.plot(subject=f'fsaverage', **surfer_kwargs) 

    # Save the image if an output file is specified
    if output_file:
        brain.save_image(output_file)
        
    # Start the event loop
    # app.exec_()
    brain.close()

    return brain
    
# --------------------------------------------------------------------------

def source_localisation(inverse_operator, epochs, method, time_viewer=False, output_file=None, view='lateral'):
    
    evoked = epochs.average()

    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stc = apply_inverse(evoked, inverse_operator, lambda2, method, pick_ori=None)
    
    mean_data = np.mean(stc.data, axis=1)
    stc.data = mean_data[:, np.newaxis]

    if output_file is not None:
        plot_source_estimate(stc, subjects_dir, hemisphere="lh", time_viewer=time_viewer, output_file=output_file + "_lh.png", view=view)
        plot_source_estimate(stc, subjects_dir, hemisphere="rh", time_viewer=time_viewer, output_file=output_file + "_rh.png", view=view)
    else:
        plot_source_estimate(stc, subjects_dir, hemisphere="lh", time_viewer=time_viewer, view=view)
        plot_source_estimate(stc, subjects_dir, hemisphere="rh", time_viewer=time_viewer, view=view)

    return stc

# --------------------------------------------------------------------------

def create_colorbar_image(output_path, colormap='viridis', orientation='vertical', width=100, height=400, vmin=0, vmax=1, ticks=None):
    # Create a figure and a colorbar
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    fig.subplots_adjust(right=0.5)

    # Create a colorbar with the specified colormap and range
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation=orientation)

    # Set custom ticks if provided
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{tick:.2f}' for tick in ticks])
    else:
        num_ticks = 5  # Default to 11 ticks
        ticks = np.linspace(vmin, vmax, num_ticks)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{tick:.2f}' for tick in ticks])

    # Save the colorbar image
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# --------------------------------------------------------------------------

if __name__ == "__main__":
    
    b_source_localisation = False
    b_cut_images = False
    b_concat_images = True
    
    result_path = "results"
    methods = ['sLORETA'] # , 'eLORETA', 'dSPM', 'MNE']
    ds_name = "KUL"
    class_labels = [0,1]
    cluster_ids = [1,2]
    data_domains = ['X_time']
                       
    font_path = r"C:\Users\hendr\Documents\GitProjects\SourceLocalisation\dejavu-sans\DejaVuSans.ttf"
    
    # ---------------------------------------------------------------------------------------
    
    if b_source_localisation:
        
        for method in methods:
            print(f"Method: {method}")
        
            for data_domain in ['time']:
                print(f"Data: {data_domain}")
                
                for class_label in class_labels:
                    print(f"CLASS_LABEL: {class_label}")
                    
                    for cluster_id in cluster_ids:
                        print(f"CLUSTER: {cluster_id}")
                        
                        inverse_operator_path = f"{result_path}/inverse_operator_{ds_name}_class_{class_label}_cluster_{cluster_id}_data_{data_domain}.fif"

                        # load epochs object
                        epochs_X = utils.get_cluster_data(cluster_id=cluster_id, class_label=class_label, dataset=ds_name, domain=f"X_{data_domain}")
                        epochs_X.set_eeg_reference(projection=True)

                        inverse_operator = utils.load_inverse_operator(epochs_X, inverse_operator_path, overwrite=False)
                                    
                        epochs_R = utils.get_cluster_data(cluster_id=cluster_id, class_label=class_label, dataset=ds_name, domain=f"R_{data_domain}")
                        epochs_R.set_eeg_reference(projection=True)
                        
                        for view in ['lateral', 'rostral', 'caudal']:  
                            
                            file_path = f"images/orig/X_method_{method}_class_{class_label}_clstr_{cluster_id}_{view}"
                            
                            stc = source_localisation(inverse_operator, epochs_X, method, output_file=file_path, view=view) 

                            file_path_min_max_X = f"images/orig/X_method_{method}_class_{class_label}_clstr_{cluster_id}_min_max_tuple"
                            joblib.dump((stc.data.min(), stc.data.max()), file_path_min_max_X)
                            
                            # ------------------------    
                            
                            epochs_R._data = np.abs(epochs_R._data) 
                            # # epochs_R._data[epochs_R._data < 0] = 0
                            
                            file_path = f"images/orig/R_method_{method}_class_{class_label}_clstr_{cluster_id}_{view}"
                            stc = source_localisation(inverse_operator, epochs_R, method, output_file=file_path, view=view) 
       
                            file_path_min_max_R = f"images/orig/R_method_{method}_class_{class_label}_clstr_{cluster_id}_min_max_tuple"
                            joblib.dump((stc.data.min(), stc.data.max()), file_path_min_max_R)

    # ---------------------------------------------------------------------------------------
    
    if b_cut_images:
        
        width = 600
        height = 600

        bottom = (120.0/600.0) * height
        top = (140.0/600.0) * height

        for method in methods:
            print(f"Method: {method}")
        
            for data_domain in ['time']:
                print(f"Data: {data_domain}")
                
                for class_label in class_labels:
                    print(f"CLASS_LABEL: {class_label}")
                    
                    for data in ['X','R']:
                    
                        for cluster_id in cluster_ids:
                            print(f"CLUSTER: {cluster_id}")
                            
                            path_input = f"images/orig/{data}_method_{method}_class_{class_label}_clstr_{cluster_id}_lateral_lh.png"
                            # path_output = f"images/cut/{data}_method_{method}_class_{class_label}_clstr_{cluster_id}_colorbar.png"
                            # top_ = 800
                            # bottom_ = 50
                            # left = (80.0/600.0) * width
                            # crop_image_pillow(path_input, path_output, left=left, right=left, top=top_, bottom=bottom_)

                            for hemis in ['lh', 'rh']:                            
                                for view in ['lateral', 'rostral', 'caudal']: 
                                    
                                    if view == "lateral":
                                        left = (25.0/600.0) * width
                                    elif view == "rostral":
                                        left = (250.0/600.0) * width
                                    else:
                                        left = (250.0/600.0) * width
                                    
                                    path_input = f"images/orig/{data}_method_{method}_class_{class_label}_clstr_{cluster_id}_{view}_{hemis}.png"
                                    path_output = f"images/cut/{data}_method_{method}_class_{class_label}_clstr_{cluster_id}_{view}_{hemis}.png"
                        
                                    crop_image_pillow(path_input, path_output, left=left, right=left, top=top, bottom=bottom)

    # ---------------------------------------------------------------------------------------
    
    if b_concat_images:
        
        for method in methods:
            print(f"Method: {method}")
        
            for data_domain in ['time']:
                print(f"Data: {data_domain}")
                
                for class_label in class_labels:
                    print(f"CLASS_LABEL: {class_label}")
                    
                    # for data in ['X','R']:
                    
                    for cluster_id in cluster_ids:
                        print(f"CLUSTER: {cluster_id}")
                        
                        file_path_min_max_X = f"images/orig/X_method_{method}_class_{class_label}_clstr_{cluster_id}_min_max_tuple"
                        file_path_min_max_R = f"images/orig/R_method_{method}_class_{class_label}_clstr_{cluster_id}_min_max_tuple"

                        min_max_X = joblib.load(file_path_min_max_X)
                        min_max_R = joblib.load(file_path_min_max_R)

                        paths_input_X = []
                        paths_input_R = []

                        for hemis in ['lh', 'rh']:                            
                            for view in ['lateral', 'rostral', 'caudal']: 
                                paths_input_X.append(f"images/cut/X_method_{method}_class_{class_label}_clstr_{cluster_id}_{view}_{hemis}.png")
                                paths_input_R.append(f"images/cut/R_method_{method}_class_{class_label}_clstr_{cluster_id}_{view}_{hemis}.png")
                            
                        path_output = f"images/concat/method_{method}_class_{class_label}_clstr_{cluster_id}.png"
                    
                        # offset = (325, 0) 
                        offset = (0, 0) 
                        
                        concatenate_images(paths_input_X, paths_input_R, path_output, min_max_X, min_max_R, offset=offset)

                        img = Image.open(path_output)

                        ## if class_label: text = "Left Ear"
                        ## else: text ="Right Ear"
                        ## location = (img.width//2, 0)
                        ## add_text_to_image(path_output, path_output, text, location, font_path=font_path, font_size=20, font_rotation=0)

                        # text = f"Cluster {cluster_id}"
                        # location = (0, (img.height//2)-25)
                        # add_text_to_image(path_output, path_output, text, location, font_path=font_path, font_size=17, font_rotation=90)

                        # text = f"EEG"
                        # y_pos = img.height//4 # + 25
                        # location = (30, y_pos)
                        # add_text_to_image(path_output, path_output, text, location, font_path=font_path, font_size=15, font_rotation=90)

                        # text = f"Relevance"
                        # y_pos = img.height//4 + 100 
                        # location = (30, y_pos)
                        # add_text_to_image(path_output, path_output, text, location, font_path=font_path, font_size=15, font_rotation=90)

# --------------------------------------------------------------------------

