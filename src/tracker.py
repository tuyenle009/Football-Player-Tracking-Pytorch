import cv2

class Tracker:
    def _draw_ellipse(self, img, xyxy, xywh, color):
        # Unpack coordinates and dimensions
        xmin, ymin, xmax, ymax = xyxy
        xcent, ycent, width, height = xywh

        # Ellipse parameters
        center = (xcent, ymax)
        axes = (width, int(width * 0.35))
        angle, startAngle, endAngle = 0.0, -45, 240
        thickness, lineType = 2, cv2.LINE_4

        # Draw the ellipse
        img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType)
        return img

    def _draw_rectangle(self, img, xyxy, xywh, color, track_id):
        # Unpack coordinates and dimensions
        xmin, ymin, xmax, ymax = xyxy
        xcent, ycent, width, height = xywh

        # Rectangle parameters
        rect_w, rect_h = 40, 20
        x1, x2 = xcent - rect_w // 2, xcent + rect_w // 2
        y1, y2 = ymax - rect_h // 2 + 15, ymax + rect_h // 2 + 15
        text_pos = (x1 + 8, y1 + 15)

        # Draw the filled rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
        # Draw the track_id text
        cv2.putText(img, "{}".format(track_id), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 50, 255), 2)
        return img

    def draw_annotations(self, img, bbox, color, track_id):
        # Unpack coordinates and calculate center and size
        xmin, ymin, xmax, ymax = bbox
        xcent, ycent, width, height = (xmax + xmin) // 2, (ymax + ymin) // 2, (xmax - xmin), (ymax - ymin)

        # Draw ellipse and rectangle with text
        img = self._draw_ellipse(img, [xmin, ymin, xmax, ymax], [xcent, ycent, width, height], color)
        img = self._draw_rectangle(img, [xmin, ymin, xmax, ymax], [xcent, ycent, width, height], color, track_id)
        return img