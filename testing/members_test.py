import inspect
from sklearn.cluster import AgglomerativeClustering

m = inspect.getmembers(AgglomerativeClustering,predicate=inspect.ismethod)

method_list = [func for func in dir(AgglomerativeClustering) if callable(getattr(AgglomerativeClustering, func))]
print(method_list)
