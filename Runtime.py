import timeit


def PrintRuntime(name: str, func):
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
    print(name, ':', round(stop - start, ndigits=2), 'seconds')
    return res


def Runtime(name: str, func):
    """
    Prints runtime of given function and returns its the runtime

    Parameters
    ----------
    name : str
        The name to be used when printing
    func
        Lambda expression to be timed

    Returns
    -------
    float
        The seconds it took to run the lambda expression
    """
    start = timeit.default_timer()
    func()
    stop = timeit.default_timer()
    
    runtime = round(stop - start, ndigits=2)
    print(name, ':', runtime, 'seconds')
    return runtime
