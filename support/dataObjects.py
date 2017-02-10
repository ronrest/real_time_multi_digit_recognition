from __future__ import print_function, absolute_import
from . import np
from image_processing import batch_augmentation
from support import verbose_print, verbose_print_done
import copy

# ##############################################################################
#                                                                        DATAOBJ
# ##############################################################################
class DataObj(object):
    def __init__(self, X=None, Y=None, batchsize=32):
        """ Create a Data Object.
        
        Args:
            X:          (np array) array of images
            Y:          (dict) labels dictionary containing the following keys:
                        - "digits"
                        - "N"
                        - "bboxes"
            batchsize: (int) batch size to use for this data. (default=32)
        """
        self.X = X
        self.Y = None
        self.N = None
        self.BBOX = None
        
        self._batchsize = batchsize
        
        # EXTRACT LABELS DATA FROM Y
        if Y is not None:
            self._extract_labels_from_Y_dict(Y=Y)
    
    @property
    def digit_bboxes(self):
        """Returns the bounding boxes for just the individual digits"""
        return self.BBOX[:,4:]

    @property
    def whole_bboxes(self):
        """Returns the bounding boxes for just the whole number, not the digits
        """
        return self.BBOX[:, :4]

    @property
    def n_samples(self):
        return self.X.shape[0]
    
    @property
    def batchsize(self):
        return self._batchsize
    
    def set_batchsize(self, n):
        self._batchsize = n
    
    @property
    def n_batches(self):
        """ returns the number of whole batches needed to complete an epoch.
            Note: it only counts full batches, and skips the final small
            batch if it does not multiply evenly.
        """
        return self.n_samples // self.batchsize
    
    def _extract_labels_from_Y_dict(self, Y):
        self.Y = Y["digits"]
        self.N = Y["N"]
        self.BBOX = Y["bboxes"]
    
    def steal_n_samples(self, n=1024, random=False, verbose=False):
        """ Removes n samples from the data and returns those removed samples
            as a new data object.

            You can optionaly select to steal the first n items (default) or
            a random sample of n items (random=True).
        """
        verbose_print("Stealing {} samples. ".format(n), verbose, end="")

        # SELECT WHICH INDICES TO STEAL
        if random:
            indices = np.random.choice(np.arange(self.n_samples), size=n,
                                       replace=False)
        else:
            indices = np.arange(n)
        
        # CREATE NEW DATA OBJ WITH EXTRACTED DATA
        new_data = DataObj()
        new_data.X = self.X[indices]
        new_data.Y = self.Y[indices]
        new_data.N = self.N[indices]
        new_data.BBOX = self.BBOX[indices]
        
        # REMOVE DATA USED
        self.X = np.delete(self.X, indices, axis=0)
        self.Y = np.delete(self.Y, indices, axis=0)
        self.N = np.delete(self.N, indices, axis=0)
        self.BBOX = np.delete(self.BBOX, indices, axis=0)
        
        verbose_print_done(verbose)
        return new_data

    def get_n_samples(self, n=1024, random=False, indices=None, verbose=False):
        """ Gets n number of samples and returns those samples
            as a new data object.

            You can optionaly select to get the first n items (default) or
            a random sample of n items (random=True).
        """
        verbose_print("Creating a sample of {} items".format(n), verbose, end="")
        # SELECT WHICH INDICES TO USE
        if random:
            indices = np.random.choice(np.arange(self.n_samples), size=n,
                                       replace=False)
        else:
            indices = np.arange(n)

        new_data = self.extract_items(indices=indices, deepcopy=True, verbose=False)
        verbose_print_done(verbose)

        # # CREATE NEW DATA OBJ WITH EXTRACTED DATA
        # new_data = DataObj()
        # new_data.X = self.X[indices]
        # new_data.Y = self.Y[indices]
        # new_data.N = self.N[indices]
        # new_data.BBOX = self.BBOX[indices]
        #
        # verbose_print_done(verbose)
        # return copy.deepcopy(new_data)
        return new_data
    
    def extract_items(self, indices, deepcopy=True, verbose=False):
        """ Given a list of indices, it returns a new DataObj that is a subset
            of this one, that includes only items at those indices.
            
            By default it creates a deep copy of the data object.
        
        Args:
            indices:  (list) the indices you want to keep
            deepcopy: (bool) creates a deep copy of the data
            verbose:  (bool)

        Returns:
            DataObj
        """
        verbose_print("Extracting a slice of items", verbose, end="")
        # CREATE NEW DATA OBJ WITH EXTRACTED DATA
        new_data = DataObj()
        new_data.X = self.X[indices]
        new_data.Y = self.Y[indices]
        new_data.N = self.N[indices]
        new_data.BBOX = self.BBOX[indices]
        verbose_print_done(verbose)
        
        if deepcopy:
            return copy.deepcopy(new_data)
        else:
            return new_data

    def limit_samples(self, n, verbose=False):
        """ Limits the data to a maximum of the first `n` samples. Deletes
            everything else.
            WARNING: This modifies the data object in-place.
            
        Args:
            n:  (int or None) Limit data to this number of samples.
                If None is passed, then the data remains intact, no trimming
                occurs.
            verbose: (bool)
        """
        if n is not None:
            verbose_print("Limiting data to {} samples".format(n), verbose, end="")
            self.X = self.X[:n]
            self.Y = self.Y[:n]
            self.N = self.N[:n]
            self.BBOX = self.BBOX[:n]
            verbose_print(" -- [DONE]", verbose)
        else:
            verbose_print("No limit was imposed, all data will be used. ".format(n),
                          verbose, end="\n")

    def shuffle(self, verbose=False):
        verbose_print("Shuffling the Data", verbose, end="")
        i_shuffled = np.random.permutation(self.n_samples)
        self.X = self.X[i_shuffled]
        self.Y = self.Y[i_shuffled]
        self.N = self.N[i_shuffled]
        self.BBOX = self.BBOX[i_shuffled]
        verbose_print_done(verbose)
    
    def batch_indices(self, batch_n):
        """ Given a batch number, it returns the lower, and upper indices
            needed to slice the data for the given batch.
        """
        i = batch_n * self.batchsize
        j = i + self.batchsize
        return (i,j)
    
    def create_batch(self, batch_n, augment=False, verbose=False):
        """ Given the n'th batch number, it creates a new data object with
            that batch of data.

            Optionally allows you to do data augmentation on that batch.
        """
        verbose_print("Creating a Batch of Data", verbose, end="")
        # CREATE NEW DATA OBJ WITH EXTRACTED DATA
        i,j = self.batch_indices(batch_n)
        batch_data = DataObj(batchsize=self.batchsize)
        batch_data.X = self.X[i:j]
        batch_data.Y = self.Y[i:j]
        batch_data.N = self.N[i:j]
        batch_data.BBOX = self.BBOX[i:j]
        
        if augment:
            batch_data.do_random_transforms()
            # batch_data.X, batch_data.BBOX = batch_augmentation(
            #     X=batch_data.X,
            #     bboxes=batch_data.BBOX,
            #     N=batch_data.N,
            #     crop=54,
            #     random_crop=True,
            #     brightness=0.3,
            #     contrast=0.3,
            #     blur=1,
            #     noise=10,
            #     seed=None)
        verbose_print_done(verbose)
        return batch_data
    
    def do_random_transforms(self, verbose=False):
        """ Modifies the X data by applying random transforms on the image data,
            such as cropping, shifting, brightness, contrast, blur and noise.
            
            Also updates the bounding box labels to account for these
            transformations.
            
            NOTE: the images are rescaled to 54x54
        """
        verbose_print("Randomly transform the images", verbose, end="")
        self.X, self.BBOX = batch_augmentation(
                X=self.X,
                bboxes=self.BBOX,
                N=self.N,
                crop=54,
                random_crop=True,
                brightness=0.3,
                contrast=0.3,
                blur=1,
                noise=10,
                seed=None)
        verbose_print_done(verbose)
        
    def do_center_crops(self):
        """ Modifies the X data so that it is center cropped to 54x54.
            NOTE: that this modifies the data in place. It does not return a
            copy of the data.
        """
        self.X, self.BBOX = batch_augmentation(X=self.X,
                                               bboxes=self.BBOX,
                                               N=self.N,
                                               crop=54,
                                               random_crop=False)
    
    def digit_lengths_distribution(self, ratio=False):
        """ returns a tally of digit lengths. You can optionally return the
            values as ratios (that add up to 1) instead of counts. """
        tally = np.array(np.unique(self.N, return_counts=True)).transpose()
        
        if ratio:
            tally = tally.astype(dtype=np.float32)
            tally[:, 1] = tally[:, 1] / float(self.n_samples)
        
        return tally


# ##############################################################################
#                                                                    DATAOBJECTS
# ##############################################################################
class DataObjects(object):
    """ Class for storing train, valid and test data objects in one place."""
    def __init__(self):
        pass
    
    def set_train_data(self, X, Y, batchsize=32):
        self.train = DataObj(X=X, Y=Y, batchsize=batchsize)
    
    def set_test_data(self, X, Y, batchsize=128):
        self.test = DataObj(X=X, Y=Y, batchsize=batchsize)
        self.valid.do_center_crops()

    def set_valid_data(self, n=1024, random=True, batchsize=128, verbose=False):
        verbose_print("Creating Validation Data", verbose, end="")
        self.valid = self.train.steal_n_samples(n=n, random=random)
        self.valid.do_center_crops()
        self.valid.set_batchsize(n=batchsize)
        verbose_print_done(verbose)
    
    def set_train_eval_data(self, n=1024, random=False, random_transforms=True, batchsize=128, verbose=False):
        verbose_print("Creating Train Evaluation Data", verbose, end="")
        self.train_eval = self.train.get_n_samples(n=n, random=random)
        
        if random_transforms:
            self.train_eval.do_random_transforms()
        
        self.train_eval.set_batchsize(n=batchsize)
        verbose_print_done(verbose)

