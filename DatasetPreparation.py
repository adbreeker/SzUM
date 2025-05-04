import os
import shutil
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

def CopySplitDataFromDir(source_dir, output_dir, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20):
    
    #remove and recreate split dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed existing '{output_dir}/'")
    os.makedirs(output_dir, exist_ok=True)

    classes = [cls for cls in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, cls))]

    #coping and spliting
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = os.listdir(cls_path)
        images = [img for img in images if os.path.isfile(os.path.join(cls_path, img))]

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        #split images to sets
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        counters = {
            'train': 1,
            'val': 1,
            'test': 1
        }

        #create splits and classes directories and copy splited images
        for split_name, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_class_dir, exist_ok=True)

            for img in split_imgs:
                src = os.path.join(cls_path, img)
                ext = os.path.splitext(img)[1].lower()  #keep extension
                new_name = f"{cls}_{counters[split_name]}{ext}"
                counters[split_name] += 1

                dst = os.path.join(split_class_dir, new_name)
                shutil.copyfile(src, dst)

    print(f"{output_dir} complete")

def CreateSplit3FromSplit2(split3_dir, split2_dir):
    # Remove and recreate split3_dir
    if os.path.exists(split3_dir):
        shutil.rmtree(split3_dir)
        print(f"Removed existing '{split3_dir}/'")
    shutil.copytree(split2_dir, split3_dir)
    print(f"Copied contents from '{split2_dir}' to '{split3_dir}'")

    train_dir = os.path.join(split3_dir, 'train')
    val_dir = os.path.join(split3_dir, 'val')

    # For each class, replace val with part of train (copy only, no deletion from train)
    class_names = os.listdir(val_dir)
    for cls in class_names:
        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)

        train_imgs = os.listdir(train_cls_dir)
        val_imgs = os.listdir(val_cls_dir)
        val_imgs_count = len(val_imgs)

        if val_imgs_count == 0:
            print(f"Skipping class '{cls}' due to no val images")
            continue

        if len(train_imgs) < val_imgs_count:
            print(f"Not enough train images to copy for class '{cls}' (train: {len(train_imgs)}, val target: {val_imgs_count})")
            continue

        # Clear val directory
        for file in val_imgs:
            os.remove(os.path.join(val_cls_dir, file))

        # Copy first n_val images from train to val
        for i in range(val_imgs_count):
            src = os.path.join(train_cls_dir, train_imgs[i])
            dst = os.path.join(val_cls_dir, train_imgs[i])
            shutil.copyfile(src, dst)

        print(f"Copied {val_imgs_count} train images to val for class '{cls}'")

    print(f"{split3_dir} ready with copied validation set")

    

def StandardizeAndAugmentData(source_dir, img_size=(128, 128), augmented_copies = 1, extension = ".jpg"):
    #augmentation for training 
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=[-10, 10],
        height_shift_range=[-10, 10],
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 2.0]
    )

    #augmentation for test/val (no augmentation)
    simple_datagen = ImageDataGenerator()

    def ProcessAndOverride(subdir, datagen, augment=False):
        full_path = os.path.join(source_dir, subdir)
        class_names = os.listdir(full_path)

        #iterate through classes subdirectories
        for class_name in class_names:
            class_path = os.path.join(full_path, class_name)
            images = os.listdir(class_path)
            images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

            processed_images = []

            #standarize (and if True - augment) images
            for idx, img in enumerate(images):
                img_path = os.path.join(source_dir, subdir, class_name, img)
                original_img = load_img(img_path, target_size=img_size)
                x = img_to_array(original_img)
                x = np.expand_dims(x, axis=0)
                batch = []

                if augment:
                    gen = datagen.flow(x, batch_size=1)
                    for i in range(augmented_copies):
                        batch.append(gen.next()[0])
                
                processed_images.append([x, batch])


            for i, (original, augmented_batch) in enumerate(processed_images):
                #save original
                output_img = array_to_img(np.squeeze(original, axis=0))
                new_name = f"{class_name}_{i + 1}.jpg"
                output_img.save(os.path.join(class_path, new_name))
            
                #save augmented copies
                for j, aug_img in enumerate(augmented_batch):
                    output_img = array_to_img(aug_img)
                    new_name = f"{class_name}_{i + 1}_{j + 1}{extension}"
                    output_img.save(os.path.join(class_path, new_name)) 
            
            #remove not matching extensions 
            for file in os.listdir(class_path):
                if not file.lower().endswith(extension):
                    os.remove(os.path.join(class_path, file))

            print(f"{class_name} in {class_path} finished")

    #process all subsets
    ProcessAndOverride('train', train_datagen, augment=True)
    ProcessAndOverride('val', simple_datagen)
    ProcessAndOverride('test', simple_datagen)

    print(f"Images normalized, augmented, and saved in {source_dir}")

#changed after part 2 presentation - now using rgb instead of greyscale
def LoadDatasetWithNormalization(source_dir, batch_size=32, target_size=(128, 128)): 
    print(f"Loading data from {source_dir}")
    datagen = ImageDataGenerator(rescale=1./255)
    dataset = datagen.flow_from_directory(
        source_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True
    )
    return dataset

#run ----------------------------------------------------------------------------------------------------------- run

if __name__ == "__main__":
    original_data_dir = './OriginalDataset'
    split1_dir = './Split1'
    split2_dir = './Split2'
    split3_dir = './Split3'

    CopySplitDataFromDir(original_data_dir, split1_dir)
    CopySplitDataFromDir(original_data_dir, split2_dir)
    CopySplitDataFromDir(original_data_dir, split3_dir)

    StandardizeAndAugmentData(split2_dir, augmented_copies=9)

    CreateSplit3FromSplit2(split3_dir, split2_dir)

    print("Dataset preparation finished")
