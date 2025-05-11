import time

class Timer:
    """
    Timer utility class to measure execution time of code blocks or operations.
    """
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def start(self):
        """
        Starts the timer.
        """
        self.start_time = time.time()
        self.elapsed_time = 0.0
        print("Timer started...")

    def stop(self):
        """
        Stops the timer and calculates the elapsed time.
        """
        if self.start_time is None:
            raise ValueError("Timer not started. Use start() before stop().")
        
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"Timer stopped. Elapsed time: {self.elapsed_time:.4f} seconds")

    def reset(self):
        """
        Resets the timer.
        """
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0.0
        print("Timer reset.")

    def get_elapsed_time(self):
        """
        Returns the elapsed time in seconds.
        """
        if self.start_time is None:
            raise ValueError("Timer not started. Use start() before getting elapsed time.")
        
        if self.end_time is not None:
            return self.elapsed_time
        else:
            return time.time() - self.start_time

def print_elapsed_time(timer: Timer):
    """
    Prints the elapsed time using the Timer class.
    """
    elapsed_time = timer.get_elapsed_time()
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
