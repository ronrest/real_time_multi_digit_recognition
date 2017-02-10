import time

################################################################################
#                                                                          TIMER
################################################################################
class Timer():
    """Creates a timer object to measure elapsed time"""
    t0 = None
    
    def start(self):
        """Starts the timer"""
        self.t0 = time.time()
        return self.t0
    
    def stop(self):
        """Stops the timer, and returns the lapsed time. Resets t0"""
        assert self.t0 is not None, "Timer has not been started yet"
        lapsed_time = time.time() - self.t0
        self.t0 = None
        return lapsed_time
    
    def lap(self):
        """Returns the elapsed time so far, but does not stop the timer"""
        assert self.t0 is not None, "Timer has not been started yet"
        lapsed_time = time.time() - self.t0
        return lapsed_time

