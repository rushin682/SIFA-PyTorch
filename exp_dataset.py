import time
from options.train_options import TrainOptions
from data import create_dataset

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    opt.num_classes = dataset.class_count() # Not Sure if this will work, but I hope it does.
    print('The number of training images = %d' % dataset_size)
    print('The number of classes = %d' % num_classes)

    iter_data_time = time.time()

    for i, data in enumerate(dataset):
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size

        print("batch: %d" %(i))
        print("Sample: ", data[0].S_paths) #check for semantic errors
         # Add Tensorboard functionality to get a view of the batch images
