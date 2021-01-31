import numpy as np
from PIL import Image, ImageDraw
from utils.utils import resize_image


def point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False


def point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge):
    nth = -1
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
    for ith in range(2):
        quad_xy_list = np.concatenate((
            np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
            np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
        p_min = np.amin(quad_xy_list, axis=0)
        p_max = np.amax(quad_xy_list, axis=0)
        if point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
            if nth == -1:
                nth = ith
            else:
                nth = -1
                break
    return nth


def shrink(xy_list, shrink_ratio=0.2, epsilon=1e-4):
    if shrink_ratio == 0.0:
        return xy_list, xy_list
    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
    diff = np.concatenate((diff_1to3, diff_4), axis=0)
    dis = np.sqrt(np.sum(np.square(diff), axis=-1))
    # determine which are long or short edges
    long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))
    short_edge = 1 - long_edge
    # cal r length array
    r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
    # cal theta array
    diff_abs = np.abs(diff)
    diff_abs[:, 0] += epsilon
    theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])
    # shrink two long edges
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, shrink_ratio)
    shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, shrink_ratio)
    # shrink two short edges
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, shrink_ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, shrink_ratio)
    return temp_new_xy_list, new_xy_list, long_edge


def shrink_edge(xy_list, new_xy_list, edge, r, theta, shrink_ratio=0.2):
    if shrink_ratio == 0.0:
        return
    start_point = edge
    end_point = (edge + 1) % 4
    long_start_sign_x = np.sign(
        xy_list[end_point, 0] - xy_list[start_point, 0])
    new_xy_list[start_point, 0] = \
        xy_list[start_point, 0] + \
        long_start_sign_x * shrink_ratio * r[start_point] * np.cos(theta[start_point])
    long_start_sign_y = np.sign(
        xy_list[end_point, 1] - xy_list[start_point, 1])
    new_xy_list[start_point, 1] = \
        xy_list[start_point, 1] + \
        long_start_sign_y * shrink_ratio * r[start_point] * np.sin(theta[start_point])
    # long edge one, end point
    long_end_sign_x = -1 * long_start_sign_x
    new_xy_list[end_point, 0] = \
        xy_list[end_point, 0] + \
        long_end_sign_x * shrink_ratio * r[end_point] * np.cos(theta[start_point])
    long_end_sign_y = -1 * long_start_sign_y
    new_xy_list[end_point, 1] = \
        xy_list[end_point, 1] + \
        long_end_sign_y * shrink_ratio * r[end_point] * np.sin(theta[start_point])


def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


def reorder_vertexes(xy_list, epsilon=1e-4):
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) / (xy_list[index, 0] - xy_list[first_v, 0] + epsilon)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
            xy_list[second_v, 0] - xy_list[fourth_v, 0] + epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


def ignore_tags_filter(anno_list, ignore_tags):
    anno_list_f = []
    for item in anno_list:
        flag = False
        for tag in ignore_tags:
            if tag in item:
                flag = True
                break
        if not flag:
            anno_list_f.append(item)
    return anno_list_f


def adjust_label(anno_list, scale_ratio_w, scale_ratio_h, ignore_tags):
    anno_list = ignore_tags_filter(anno_list, ignore_tags)
    xy_list_array = np.zeros((len(anno_list), 4, 2))
    for anno, i in zip(anno_list, range(len(anno_list))):
        anno_colums = anno.strip().split(',')
        anno_array = np.array(anno_colums)
        xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
        xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
        xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
        xy_list = reorder_vertexes(xy_list)
        xy_list_array[i] = xy_list
    return xy_list_array


def get_text_polys(anno_list, ignore_tags):
    anno_list = ignore_tags_filter(anno_list, ignore_tags)
    gt = []
    for anno, i in zip(anno_list, range(len(anno_list))):
        anno_colums = anno.strip().split(',')
        anno_array = np.array(anno_colums)
        xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
        gt.append({'points': xy_list, 'ignore': False})
    return gt


def draw_gt(draw, xy_list, shrink_xy_list, shrink_1, long_edge):
    draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
               tuple(xy_list[2]), tuple(xy_list[3]),
               tuple(xy_list[0])],
              width=2, fill='green')
    draw.line([tuple(shrink_xy_list[0]),
               tuple(shrink_xy_list[1]),
               tuple(shrink_xy_list[2]),
               tuple(shrink_xy_list[3]),
               tuple(shrink_xy_list[0])],
              width=2, fill='blue')
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
    for q_th in range(2):
        draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                   tuple(shrink_1[vs[long_edge][q_th][1]]),
                   tuple(shrink_1[vs[long_edge][q_th][2]]),
                   tuple(xy_list[vs[long_edge][q_th][3]]),
                   tuple(xy_list[vs[long_edge][q_th][4]])],
                  width=3, fill='yellow')


def make_gt(im, anno_list, target_size=640, shrink_ratio=0.2, redution=4, shrink_side_ratio=0.6, draw_gt_quad=False, ignore_tags='###'):
    d_width, d_height = target_size, target_size
    im_height, im_width = im.height, im.width
    scale_ratio_w = d_width / im_width
    scale_ratio_h = d_height / im_height
    im = im.resize((d_width, d_height), Image.NEAREST).convert('RGB')
    xy_list_array = adjust_label(anno_list, scale_ratio_w, scale_ratio_h, ignore_tags)
    gt = np.zeros((d_height // redution, d_width // redution, 7))
    show_gt_im = im.copy()
    draw = ImageDraw.Draw(show_gt_im)

    for xy_list in xy_list_array:
        _, shrink_xy_list, _ = shrink(xy_list, shrink_ratio)
        shrink_1, _, long_edge = shrink(xy_list, shrink_side_ratio)

        if draw_gt_quad:
            draw_gt(draw, xy_list, shrink_xy_list, shrink_1, long_edge)

        p_min = np.amin(shrink_xy_list, axis=0)
        p_max = np.amax(shrink_xy_list, axis=0)
        # floor of the float
        ji_min = (p_min / redution - 0.5).astype(int) - 1
        # +1 for ceil of the float and +1 for include the end
        ji_max = (p_max / redution - 0.5).astype(int) + 3
        imin = np.maximum(0, ji_min[1])
        imax = np.minimum(d_height // redution, ji_max[1])
        jmin = np.maximum(0, ji_min[0])
        jmax = np.minimum(d_width // redution, ji_max[0])
        for i in range(imin, imax):
            for j in range(jmin, jmax):
                px = (j + 0.5) * redution
                py = (i + 0.5) * redution
                if point_inside_of_quad(px, py, shrink_xy_list, p_min, p_max):
                    gt[i, j, 0] = 1
                    line_width, line_color = 1, 'red'
                    ith = point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge)
                    vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                    if ith in range(2):
                        gt[i, j, 1] = 1
                        if ith == 0:
                            line_width, line_color = 2, 'yellow'
                        else:
                            line_width, line_color = 2, 'green'
                        gt[i, j, 2:3] = ith
                        gt[i, j, 3:5] = xy_list[vs[long_edge][ith][0]] - [px, py]
                        gt[i, j, 5:] = xy_list[vs[long_edge][ith][1]] - [px, py]
                    if draw_gt_quad:
                        draw.line([(px - 0.5 * redution, py - 0.5 * redution),
                                   (px + 0.5 * redution, py - 0.5 * redution),
                                   (px + 0.5 * redution, py + 0.5 * redution),
                                   (px - 0.5 * redution, py + 0.5 * redution),
                                   (px - 0.5 * redution, py - 0.5 * redution)],
                                  width=line_width, fill=line_color)
    return im, gt, show_gt_im
