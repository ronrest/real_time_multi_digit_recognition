from __future__ import print_function, absolute_import
from .timer import Timer
from main import np, copy

# ==============================================================================
#                                                        ASSERT_THESE_ATTRIBUTES
# ==============================================================================
def assert_these_attributes(a, name, attr):
    """ Asserts that object `a` must have all the attributes in `attr`,
        otherwise it throws an assertion error.

    Args:
        a:      (object)
        name:   (str) Name to give to your object when it prints out the error
                message.
        attr:   (iterable of strings) The names of the attributes.
    """
    
    msg = 'The {} object must contain all the following attributes: \n ' \
          '{}'.format(name, attr)
    assert all([hasattr(a, attr) for attr in attr]), msg


# ==============================================================================
#                                                                  VERBOSE_PRINT
# ==============================================================================
def verbose_print(s, verbose, end="\n"):
    """ Only prints the message in `s` if verbose is set to True.
        You can optionally set the end character, eg, to prevent going to a new
        line.
            
    Args:
        s:       (str) Message to print
        verbose: (bool) Only prints if this is set to True
        end:     (str)  The end character in message
    """
    if verbose:
        print(s, end=end)


# ==============================================================================
#                                                             VERBOSE_PRINT_DONE
# ==============================================================================
def verbose_print_done(verbose, end="\n"):
    """ prints " -- [DONE]" if verbose is set to True

    Args:
        verbose: (bool) Only prints if this is set to True
        end:     (str)(default="\n") The end character in message
    """
    if verbose:
        print(" -- [DONE]", end=end)


# ==============================================================================
#                                                                   LIMIT_STRING
# ==============================================================================
def limit_string(s, front=None, tail=None):
    """ Shortens a string to the first `front` number of characters, and `tail`
        number of characters, adding "..." in between.

        If the string length is already <= to `front` + `tail`, then it leaves
        the string unmodified.

        You can optionally select that it only returns the first `front`
        characters, by setting `tail` to None. Or the last `tail` number of
        characters by setting `front` to None. These add a "..." to the
        appropriate side to indicate if the string contains more content that
        was chopped off.

        If both `front` and `tail` are set to None, then it returns an
        unmodified string.

    Args:
        s:      (str) String to shorten
        front:  (int or None) Number of characters to use from the front
        tail:   (int or None) Number of characters to use from the tail.

    Examples:
        >>> limit_string("1234567890abcdefghij", front=5, tail=5)
        '12345...fghij'
        >>> limit_string("12345678", front=5, tail=5)
        '12345678'
        >>> limit_string("1234567890abcdefghij", front=None, tail=5)
        '...fghij'
        >>> limit_string("1234567890abcdefghij", front=5, tail=None)
        '12345...'
        >>> limit_string("1234567890abcdefghij", front=None, tail=None)
        '1234567890abcdefghij'

    Returns:
        (str)
    """
    if (front is not None) and (tail is not None):
        return s if len(s) <= (front + tail) else (s[:front] + "..." + s[-tail:])
    elif front is not None:
        return s if len(s) <= (front) else s[:front] + "..."
    elif tail is not None:
        return s if len(s) <= (tail) else "..." + s[-tail:]
    else:
        return s


# =============================================================================
#                                                                   MERGE_ARRAYS
#  =============================================================================
def merge_arrays(arrays, axis=0):
    """ Takes a list of numpy arrays, and appends all of them together along
        the axis specified.

    Args:
        arrays: (list of arrays)
        axis:   (int) the axis along which you wish to append.

    Returns:
        (numpy array) a new array where all the component arrays were appended.
    """
    merged = copy.deepcopy(arrays[0])
    for i in range(1, len(arrays)):
        im = arrays[i]
        merged = np.append(merged, im, axis=axis)
    return merged


# ==============================================================================
#                                                                   REPEAT_ARRAY
# ==============================================================================
def repeat_array(a, n, axis=0):
    """ Takes an array, and repeats it `n` number of times along the specified
        axis.

    Example:
        >>> a = np.array([[1,2],
        >>>               [3,4]])
        >>> repeat_array(a, n=3, axis=1)
        array([[1, 2, 1, 2, 1, 2],
               [3, 4, 3, 4, 3, 4]])
    Args:
        a:      (numpy array)
        n:      (int)
        axis:   (int)

    Returns:
        (numpy array)
    """
    return merge_arrays([a] * n, axis=axis)



#  =============================================================================
#                                                             DIGIT_ARRAY_TO_STR
#  =============================================================================
def digit_array_to_str(a, null=-1):
    """ Takes an array `a` of digits (and optionally the null values to ignore),
        and returns the elements stuck together to form a string:

    Example:
    >>> digit_array_to_str(np.array([10,10,5,0]), null=10)
    "50"
    """
    items = a[a != null]
    return "".join([str(c) for c in items])


#  =============================================================================
#                                                             DIGIT_ARRAY_TO_INT
#  =============================================================================
def digit_array_to_int(a, null=-1):
    """takes an array `a` of digits, (and optionally the null values to ignore),
    and puts the digits together to form an integer:

    Example:
    >>> digit_array_to_int([10,10,5,0], null=10)
    50
    """
    x = np.array(a)     # Ensure input is a numpy array
    x = x[x != null]    # Remove null values to ignore
    n = x.shape[0]      # Get length of array
    b = np.array([10**e for e in range(n-1, 0-1, -1)]) # Base 10 multipliers
    return (x * b).sum()


#  =============================================================================
#                                                  ARRAY_OF_DIGIT_ARRAYS_TO_INTS
#  =============================================================================
def array_of_digit_arrays_to_ints(a, null=-1):
    """Takes an array `a`, where each row, is an array of digits, (and
    optionally the null values to ignore), and returns each row as an integer:

    Example:
    >>> a = np.array([[10, 10, 10,  7,  4,  4],
    >>>              [10, 10, 10,  1,  2,  8],
    >>>              [10, 10, 10, 10,  1,  6]])
    >>>> array_of_digit_arrays_to_ints(a, null=10)
    array([744, 128,  16])
    """
    return np.apply_along_axis(digit_array_to_int, axis=1, arr=a, null=10)


#  =============================================================================
#                                                   ARRAY_OF_DIGIT_ARRAYS_TO_STR
#  =============================================================================
def array_of_digit_arrays_to_str(a, null=-1):
    """Takes an array `a`, where each row, is an array of digits, (and
    optionally the null values to ignore), and returns each row as a string:

    Example:
    >>> a = np.array([[10, 10, 10,  10, 7,  4],
    >>>              [10, 10, 10,  1,  2,  8],
    >>>              [10, 10, 10, 10,  1,  6]])
    >>> array_of_digit_arrays_to_str(a, null=10)
    array(['74', '128', '16'], dtype=object)
    """
    new = np.empty(shape=[a.shape[0]], dtype=np.object)
    for i in range(a.shape[0]):
        new[i] = digit_array_to_str(a[i], null=null)
    return new


#  =============================================================================
#                                                                  PRINT_HEADERS
#  =============================================================================
def print_headers(title, border="=", align="left", width=60, verbose=True):
    """ Prints out a pretty section header on the terminal. Specify the
        title you want to use, the border decoration, the title text
        alignment and the width you want it to take up.
    
    Args:
        title:  (str) The title you want to use
        border: (str) The character you want to use to decorate the title
        align:  (str) The title text alignment to use:
                - "left"
                - "right"
                - "center"
                If none of these options are provided, then it uses left
                alignment by default.
        width:  (int) How wide to make the title
        verbose: (bool) set to False if you want it to not print out anything.
                 This sounds pointless, but is useful for when you only want
                 headers to be printed out when verbose flag in your code is
                 set to True.
                 
    Examples:
        >>> print_headers("Section 1", border="-", align="center", width=20)
        --------------------
             Section 1
        --------------------
    """
    border = (border * width) + "\n"
    
    if align=="right":
        title = title.rjust(width) + "\n"
    elif align=="center":
        title = title.rjust(int(width/2 + len(title)/2)) + "\n"
    else:
        title = title + "\n"

    print(border+title+border, end="")


