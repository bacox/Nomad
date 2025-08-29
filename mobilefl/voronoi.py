import numpy as np
from scipy.spatial import Voronoi
class VoronoiMap:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.x_min = []
        self.x_max = []
        self.y_min = []
        self.y_max = []
        self.vertices = None
        self.area_for_server = []
    def buildVoronoi(self):
        vor = Voronoi(self.coordinates)
        margin = 0.1 * np.ptp(vor.points, axis=0)
        xy_min = vor.points.min(axis=0) - margin
        xy_max = vor.points.max(axis=0) + margin
        self.x_min = xy_min[0]
        self.x_max = xy_max[0]
        self.y_min = xy_min[1]
        self.y_max = xy_max[1]
        vertices, ridge_vertices = [], []
        new_dict = []
        for k, v in vor.ridge_dict.items():
            if -1 in v:
                new_dict.append(list(k))
        for ver_rela in vor.ridge_vertices:
            if -1 in ver_rela:
                temp = [0, 1]
                temp.pop(ver_rela.index(-1))
                if self.judge_frame(vor.vertices[ver_rela[temp[0]]]):
                    barr_ind = new_dict.pop(0)
                    barr = [vor.points[barr_ind[0]], vor.points[barr_ind[1]]]
                    f_ray = self.ray(
                        barr, vor.vertices[ver_rela[temp[0]]]
                    )  
                    inter = self.ray_inter_point(
                        f_ray, vor.vertices[ver_rela[temp[0]]], barr
                    )
                    if vor.vertices[ver_rela[temp[0]]].tolist() not in vertices:
                        vertices.append(vor.vertices[ver_rela[temp[0]]].tolist())
                    start = vertices.index(vor.vertices[ver_rela[temp[0]]].tolist())
                    vertices.append(inter)
                    end = vertices.index(vertices[-1])
                    ridge_vertices.append([start, end])
                else:
                    new_dict.pop(0)
                    continue
            else:
                if self.judge_frame(vor.vertices[ver_rela[0]]) and self.judge_frame(
                    vor.vertices[ver_rela[1]]
                ):
                    if vor.vertices[ver_rela[0]].tolist() not in vertices:
                        vertices.append(vor.vertices[ver_rela[0]].tolist())
                    if vor.vertices[ver_rela[1]].tolist() not in vertices:
                        vertices.append(vor.vertices[ver_rela[1]].tolist())
                    start = vertices.index(vor.vertices[ver_rela[0]].tolist())
                    end = vertices.index(vor.vertices[ver_rela[1]].tolist())
                    ridge_vertices.append([start, end])
                elif not (
                    self.judge_frame(vor.vertices[ver_rela[0]])
                    or self.judge_frame(vor.vertices[ver_rela[1]])
                ):
                    continue
                else:
                    inter = self.intersection(
                        vor.vertices[ver_rela[0]], vor.vertices[ver_rela[1]]
                    )
                    if self.judge_frame(vor.vertices[ver_rela[0]]):
                        if vor.vertices[ver_rela[0]].tolist() not in vertices:
                            vertices.append(vor.vertices[ver_rela[0]].tolist())
                            start = vertices.index(vor.vertices[ver_rela[0]].tolist())
                        else:
                            start = vertices.index(vor.vertices[ver_rela[0]].tolist())
                    else:
                        if vor.vertices[ver_rela[1]].tolist() not in vertices:
                            vertices.append(vor.vertices[ver_rela[1]].tolist())
                            start = vertices.index(vor.vertices[ver_rela[1]].tolist())
                        else:
                            start = vertices.index(vor.vertices[ver_rela[1]].tolist())
                    vertices.append(inter)
                    end = vertices.index(vertices[-1])
                    ridge_vertices.append([start, end])
        vertices = np.array(vertices)
        ridge_vertices = np.array(ridge_vertices)
        def GeneralEquation(x1, y1, x2, y2):
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            return A, B, C
        def GetIntersectPointofLines(x1, y1, x2, y2, x3, y3, x4, y4):
            A1, B1, C1 = GeneralEquation(x1, y1, x2, y2)
            A2, B2, C2 = GeneralEquation(x3, y3, x4, y4)
            m = A1 * B2 - A2 * B1
            if m == 0:
                print("无交点")
            else:
                x = (C2 * B1 - C1 * B2) / m
                y = (C1 * A2 - C2 * A1) / m
            return x, y
        wrong_value = vertices[2]
        from_value = vertices[0]
        vertices[2] = GetIntersectPointofLines(
            wrong_value[0],
            wrong_value[1],
            from_value[0],
            from_value[1],
            self.x_min,
            self.y_max,
            self.x_min,
            self.y_min,
        )
        self.vertices = vertices
        self.area_for_server.append(
            [
                vertices[0],
                vertices[3],
                vertices[5],
                [self.x_min, self.y_min],
                vertices[2],
            ]
        )
        self.area_for_server.append(
            [vertices[3], vertices[4], vertices[6], vertices[7], vertices[5]]
        )
        self.area_for_server.append(
            [vertices[6], vertices[7], [self.x_max, self.y_min], vertices[9]]
        )
        self.area_for_server.append(
            [vertices[0], vertices[1], vertices[8], vertices[4], vertices[3]]
        )
        self.area_for_server.append(
            [
                vertices[0],
                vertices[2],
                [self.x_min, self.y_max],
                [self.x_max, self.y_max],
                vertices[1],
            ]
        )
        self.area_for_server.append(
            [vertices[4], vertices[6], vertices[9], vertices[8]]
        )
    def judge_frame(self, point):
        x, y = point[0], point[1]
        flag = (
            x >= self.x_min and x <= self.x_max and y >= self.y_min and y <= self.y_max
        )
        return flag
    def judge_inter(self, A, B, C, D):
        def vector(piont_1, point_2):  
            return point_2 - piont_1
        def vector_product(vec_1, vec_2):  
            return vec_1[0] * vec_2[1] - vec_2[0] * vec_1[1]
        AC = vector(A, C)
        AD = vector(A, D)
        BC = vector(B, C)
        BD = vector(B, D)
        CA = vector(C, A)
        CB = vector(C, B)
        DA = vector(D, A)
        DB = vector(D, B)
        return (vector_product(AC, AD) * vector_product(BC, BD) <= 0) and (
            vector_product(CA, CB) * vector_product(DA, DB) <= 0
        )
    def intersection(self, point_1, point_2):
        point_bound = [
            [[self.x_min, self.y_min], [self.x_max, self.y_min]],
            [[self.x_max, self.y_min], [self.x_max, self.y_max]],
            [[self.x_min, self.y_max], [self.x_max, self.y_max]],
            [[self.x_min, self.y_max], [self.x_min, self.y_min]],
        ]
        for line in point_bound:
            if self.judge_inter(point_1, point_2, line[0], line[1]):
                x1, x2, x3, x4 = point_1[0], point_2[0], line[0][0], line[1][0]
                y1, y2, y3, y4 = point_1[1], point_2[1], line[0][1], line[1][1]
                px = (
                    (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
                ) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
                py = (
                    (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
                ) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
                return [px, py]
    def ray(self, barr, piont):
        def medLine(x1, y1, x2, y2):
            A = 2 * (x2 - x1)
            B = 2 * (y2 - y1)
            C = x1**2 - x2**2 + y1**2 - y2**2
            return A, B, C
        A, B, C = medLine(barr[0][0], barr[0][1], barr[1][0], barr[1][1])
        return [A, B, C]
    def ray_inter_point(self, ray, point, barr):
        point_bound = [
            [[self.x_min, self.y_min], [self.x_max, self.y_min]],
            [[self.x_max, self.y_min], [self.x_max, self.y_max]],
            [[self.x_min, self.y_max], [self.x_max, self.y_max]],
            [[self.x_min, self.y_max], [self.x_min, self.y_min]],
        ]
        list_inter = []
        for bound in point_bound:
            sPoint, ePoint = bound[0], bound[1]
            line = [
                sPoint[1] - ePoint[1],
                ePoint[0] - sPoint[0],
                sPoint[0] * ePoint[1] - ePoint[0] * sPoint[1],
            ]
            a0, b0, c0 = ray[0], ray[1], ray[2]
            a1, b1, c1 = line[0], line[1], line[2]
            D = a0 * b1 - a1 * b0
            x = (b0 * c1 - b1 * c0) / D
            y = (a1 * c0 - a0 * c1) / D
            list_inter.append([x, y])
        for i in list_inter:
            if self.judge_inter(
                point, np.array(i), barr[0].tolist(), barr[1].tolist()
            ) and self.judge_frame(np.array(i)):
                return i
