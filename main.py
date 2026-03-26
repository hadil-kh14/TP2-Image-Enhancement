import sys
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap

qtcreator_file = "design2.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(DesignWindow, self).__init__()
        self.setupUi(self)

        self.image_gray = None

        # Connexions boutons
        self.Browse.clicked.connect(self.get_image)
        self.Apply.clicked.connect(self.show_ImgHistEqualized)
        self.Validate_1.clicked.connect(self.show_ImgThresholding)
        self.Validate_2.clicked.connect(self.show_ImgFiltered)
        self.Validate_3.clicked.connect(self.show_ImgAugmented)

    # -------------------------------------------------
    # Utilitaire affichage image
    def makeFigure(self, path, widget):
        pixmap = QPixmap(path)
        widget.setPixmap(pixmap)
        widget.setScaledContents(True)

    # -------------------------------------------------
    # Chargement image + histogramme original
    def get_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Choisir une image", "", "Images (*.png *.jpg *.jpeg)"
        )

        if file_name:
            image = cv2.imread(file_name)
            self.image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite("Original_Image.png", self.image_gray)
            self.makeFigure("Original_Image.png", self.OriginalImg)

            self.show_HistOriginal()

    def show_HistOriginal(self):
        hist = cv2.calcHist([self.image_gray], [0], None, [256], [0, 256])

        plt.figure()
        plt.plot(hist)
        plt.title("Original Histogram")
        plt.xlabel("Intensité")
        plt.ylabel("Pixels")
        plt.savefig("Original_Histogram.png")
        plt.close()

        self.makeFigure("Original_Histogram.png", self.OriginalHist)

    # -------------------------------------------------
    # Egalisation histogramme
    def show_ImgHistEqualized(self):
        img_eq = cv2.equalizeHist(self.image_gray)

        cv2.imwrite("Equalized_Image.png", img_eq)
        self.makeFigure("Equalized_Image.png", self.EqualizedImg)

        hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

        plt.figure()
        plt.plot(hist_eq)
        plt.title("Equalized Histogram")
        plt.savefig("Equalized_Histogram.png")
        plt.close()

        self.makeFigure("Equalized_Histogram.png", self.EqualizedHist)

    # -------------------------------------------------
    # Seuillage
    def show_ImgThresholding(self):
        if self.Binary.isChecked():
            _, thresh = cv2.threshold(self.image_gray, 120, 255, cv2.THRESH_BINARY)
        else:
            _, thresh = cv2.threshold(self.image_gray, 0, 255, cv2.THRESH_OTSU)

        cv2.imwrite("Thresholding_Image.png", thresh)
        self.makeFigure("Thresholding_Image.png", self.ThresholdingImg)

    # -------------------------------------------------
    # Filtrage
    def show_ImgFiltered(self):
        if self.Mean.isChecked():
            filtered = cv2.blur(self.image_gray, (11, 11))
        elif self.Gaussian.isChecked():
            filtered = cv2.GaussianBlur(self.image_gray, (15, 15), 10)
        else:
            filtered = cv2.medianBlur(self.image_gray, 13)

        cv2.imwrite("Filtered_Image.png", filtered)
        self.makeFigure("Filtered_Image.png", self.FilteredImg)

    # -------------------------------------------------
    # Opérations géométriques
    def show_ImgAugmented(self):
        h, w = self.image_gray.shape

        if self.Rotation.isChecked():
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 45, 1.0)
            result = cv2.warpAffine(self.image_gray, M, (w, h))

        elif self.Extraction.isChecked():
            result = self.image_gray[0:h//2, 0:w//2]

        else:
            scale = random.uniform(1.5, 4.0)
            new_w, new_h = int(w * scale), int(h * scale)
            zoomed = cv2.resize(self.image_gray, (new_w, new_h),
                                interpolation=cv2.INTER_CUBIC)

            x = (new_w - w) // 2
            y = (new_h - h) // 2
            result = zoomed[y:y+h, x:x+w]

        cv2.imwrite("Augmented_Image.png", result)
        self.makeFigure("Augmented_Image.png", self.AugmentedImg)


# -------------------------------------------------
# Main
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())
