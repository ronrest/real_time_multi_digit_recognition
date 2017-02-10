import os
from . import maybe_mkdir

# ##############################################################################
#                                                                       PATHSOBJ
# ##############################################################################
class PathsObj(object):
    """ Creates a paths object.
    """
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, val):
        setattr(self, key, val)
    
    def add(self, attr, path, root=None, create_dir=False):
        """ Adds a new path attribute. Specify the attribute name `attr`, the
            `path` and optionally, if the `path` is replative to an existing
            path attribute in the object, you can specify that attribute name
            as the `root`.

        Example:
            >>> p = PathsObj()
            >>> p.add("data_dir", "/tmp/mydata")
            >>> p.add("Xtrain", "X_train.pickle", root="data_dir")
            >>> p.Xtrain
            '/tmp/mydata/X_train.pickle'

        Args:
            attr: (str) The name you want to give the attribute
            path: (str) The file path of the new file/directory
            root: (str) The name of an existing attribute that is the parent
                        directory of this new path being added.
            create_dir: (bool) If True, it actually creates this as a directory if
                        it does not already exust on the system.
        """
        if root is not None:
            setattr(self, attr, os.path.join(getattr(self, root), path))
        else:
            setattr(self, attr, path)
        if create_dir:
            maybe_mkdir(getattr(self, attr))
            
        
        