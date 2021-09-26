import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax

class TrackableObject:
    def __init__(self, objectID, keypoints, descriptors, centroid):
        self.objectID = objectID
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.centroid = centroid
    
    def update(self, keypoints, descriptors, centroid):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.centroid = centroid

class Tracker:
    def __init__(self, detector, matcher):
        # Inisiasi tracker
        self.objects = []
        self.detector = detector
        self.matcher = matcher
        self.nextObjectID = 0
    
    def get_similarity(self, kp1, ds1, kp2, ds2):
        ''' Untuk menghitung kemiripan antar 2 descriptor'''

        # Lakukan Matching
        knn_matches = self.matcher.knnMatch(ds1, ds2, 2)

        # Jika tidak ada point yang match maka knn_matches berdimensi 1 (tidak berpasang-pasangan)
        if len(knn_matches[0]) < 2:
            return 0.0

        #Filter matches using the Lowe's ratio test
        ratio_thresh = 0.6
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        
        # Hitung similiarity berdasarkan jumlah keypoint match/jumlah keypoint
        number_keypoints = 0
        if len(kp1) <= len(kp2):
            number_keypoints = len(kp1)
        else:
            number_keypoints = len(kp2)

        similiarity = len(good_matches) / number_keypoints * 100

        return similiarity
    
    def update(self, mask, img):
        '''Untuk mengupdate tracker'''

        # Threshold mask agar semua piksel bernilai 255
        _, res_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Morphology Operation untuk mengurangi noise
        kernel = np.ones((10, 10), np.uint8)
        res_mask = cv2.erode(res_mask, kernel)
        res_mask = cv2.dilate(res_mask, kernel)

        # Mencari data bounding box dan centroid dari tiap objek
        ret, labels, stats, centroid = cv2.connectedComponentsWithStats(res_mask)

        # Iterasi untuk tiap objek yang dideteksi
        current_features = []
        for stat in stats[1:]:
            left = stat[0]
            top = stat[1]
            width = stat[2]
            height = stat[3]
            centroid = [int(left+width/2), int(top+height/2)]

            # Memfilter objek yang berada diluar border deteksi
            border = 50
            if centroid[0] < border or centroid[0] > 640-border or centroid[1] < border or centroid[1] > 480-border:
                continue
            
            # Memfilter objek yang berukuran kecil
            if width*height < 400:
                continue
            
            # Cropping Image berdasarkan bounding box objek
            area_img = img[top:top+height, left:left+width]

            # Hitung descriptor dari objek menggunakan detector
            keypoints, descriptors = self.detector.detectAndCompute(area_img, None)
            if descriptors is not None:
                current_object = TrackableObject(0, keypoints, descriptors, centroid)
                current_features.append(current_object)
        
        if len(self.objects) == 0: # Cek jika sebelumnya belum ada objek yang terdeteksi (Biasanya dilakukan untuk frame pertama)
            # Jika belum ada objek append semua objek yang terdeteksi ke list objek
            for x in current_features:
                x.objectID = self.nextObjectID
                self.nextObjectID += 1
                self.objects.append(x)
        else:
            # lakukan perbandingan antara object yang dideteksi saat ini dan objects yang sebelumnya
            for i in range(len(current_features)):
                similarity = []
                for j in range(len(self.objects)):
                    kp1 = current_features[i].keypoints
                    ds1 = current_features[i].descriptors
                    kp2 = self.objects[j].keypoints
                    ds2 = self.objects[j].descriptors
                    similarity.append(self.get_similarity(kp1,ds1,kp2,ds2))

                # mencari objek paling mirip berdasarkan nilai similiarity
                max_index = argmax(similarity)

                if max(similarity) > 10: # Jika similiarity >10 artinya objek sama maka update descriptor pada object
                    keypoints = current_features[i].keypoints
                    descriptors = current_features[i].descriptors
                    centroid = current_features[i].centroid
                    self.objects[max_index].update(keypoints, descriptors, centroid)
                else: # Jika tidak maka tambahkan sebagai objek baru
                    current_features[i].objectID = self.nextObjectID
                    self.nextObjectID += 1
                    self.objects.append(current_features[i])
        
        return(self.objects)

if __name__ == "__main__":
    pass
