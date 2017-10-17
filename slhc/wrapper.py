import dslhc

class wrapper:
    def __init__(self):
        self.__world__ = dslhc.World()
        self.__point_num__ = 0
        self.__pair_num__ = 0

    def insert(self, dist):
        """
        :param dist: the input dist numpy array, input size should be the same as the current point size. 0 will be added in the end
        :return:
        """
        row = dist.tolist()
        if self.__point_num__ is not len(dist):
            raise Exception("Input distance not correct, should input %d, but receive %d" %(self.__point_num__, len(dist)))
        row.append(0)
        self.__world__.insert(row)
        self.__point_num__ += 1
        self.__pair_num__ += self.__point_num__

    def remove_rows(self, idxs):
        """
        :param idxs: the idxs list to remove
        :return:
        """
        m1 = max(idxs)
        m2 = min(idxs)
        if m2 < 0 or m1 >= self.__point_num__:
            raise Exception("The input idxs range is not correct, detected the range is %d-%d, should be 0-%d" %(m2, m1, self.__point_num__ - 1))
        self.__world__.remove_rows(idxs)
        self.__point_num__ -= len(idxs)
        self.__pair_num__ = self.__point_num__ * (1 + self.__point_num__) / 2

    def remove_one(self):
        self.remove_rows([self.__point_num__ - 1])

    def calculate(self, key):
        if key is "slhc":
            self.__world__.calculate(key)
        else:
            raise Exception("No key other than slhc is supported now!")

    def report(self, key):
        if key is not "slhc" and key is not "height" and key is  not "dist":
            raise Exception("The input key is not supported")
        return self.__world__.report(key)

    def blocks(self, delta):
        return self.__world__.blocks(delta)