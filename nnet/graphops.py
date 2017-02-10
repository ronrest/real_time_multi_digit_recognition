# ==============================================================================
#                                                                       GRAPHOPS
# ==============================================================================
class GraphOps(object):
    def __init__(self, graph, *args, **kwargs):
        """ Creates an object that allows you to access operations in a
            tensoflow graph. Simply pass the names of the ops as string
            arguments, and you can access them as attributes of the same name.

            Additional keyword arguments are interpreted as follows:
            - key: the name you want for the attribute.
            - val: the name of the operation in the graph.
                   This is particularly useful if you need to get ops whose
                   name would not make a legal python attributes, such as
                   ops that are nested within layers of scopes, such as

                        "conv1/relu"

                   For this you can specify something like:

                        "conv1relu" = "conv1/relu"

        Args:
            graph:      (tensorflow graph object)
            *args:      (strings) name of each operation from the graph you want
                        to retreive.
            **kwargs:   (strings) See the notes in the main description.

        Examples:
            > g = GraphOps(graph, "X", "Y","train")
            > sess.run(g.train, feed_dict={g.X=Data, g.Y=Labels})

            > g = GraphOps(graph, "X", "Y", convrelu="conv1/relu")
            > relu_val = sess.run(g.convrelu, feed_dict={g.X=Data, g.Y=Labels})
        """
        for arg in args:
            opname = "{}:0".format(arg)
            setattr(self, arg, graph.get_tensor_by_name(opname))
        for k, v in kwargs.items():
            opname = "{}:0".format(v)
            setattr(self, k, graph.get_tensor_by_name(opname))

