import timeit


def printRuntime(name: str, func):
    """
    Prints runtime of given function and returns its result

    Parameters
    ----------
    name : str
        The name to be used when printing
    func
        Lambda expression to be timed

    Returns
    -------
    Any
        Lambda expression result
    """
    start = timeit.default_timer()
    res = func()
    stop = timeit.default_timer()

    runtime = round(stop - start, ndigits=2)
    print(name, ':', runtime, 'seconds')

    return res


def getRuntime(name: str, func):
    """
    Prints runtime of given function and returns its result and runtime

    Parameters
    ----------
    name : str
        The name to be used when printing
    func
        Lambda expression to be timed

    Returns
    -------
    Any
        Lambda expression result
    
    float
        The seconds it took to run the lambda expression
    """
    start = timeit.default_timer()
    res = func()
    stop = timeit.default_timer()

    runtime = round(stop - start, ndigits=2)
    print(name, ':', runtime, 'seconds')
    return res, runtime
