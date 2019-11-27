import cv2


def draw_bb(img, x1, y1, x2, y2, txt, thickness=1, color=(255, 0, 0)):
    start_point = (x1, y1)
    end_point = (x2, y2)

    # bb
    img_out = cv2.rectangle(img, start_point, end_point, color, thickness)

    # txt
    cv2.putText(
        img_out,
        txt,
        (x1, y2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    return img_out







