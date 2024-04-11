import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Mat;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;
import org.opencv.core.Size;
import org.opencv.features2d.FlannBasedMatcher;

import java.util.ArrayList;
import java.util.List;

public class imageStitching {
    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat part1 = Imgcodecs.imread(".\\images\\image1.jpg");
        Mat part2 = Imgcodecs.imread(".\\images\\image2.jpg");

        int maxKeypoints = 6000;
        float contrastThreshold = 0.02f;
        float edgeThreshold = 10.0f;
        SIFT sift = SIFT.create(maxKeypoints);

        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        Mat part1Descriptors= new Mat();
        sift.detectAndCompute(part1, new Mat(), keypoints1, part1Descriptors);

        Mat outputImage1 = new Mat();
        Features2d.drawKeypoints(part1, keypoints1, outputImage1);

        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        Mat part2Descriptors= new Mat();
        sift.detectAndCompute(part2, new Mat(), keypoints2, part2Descriptors);

        Mat outputImage2 = new Mat();
        Features2d.drawKeypoints(part2, keypoints2, outputImage2);

        FlannBasedMatcher matcher = FlannBasedMatcher.create();

        List<MatOfDMatch> matchesList = new ArrayList<>();
        matcher.knnMatch(part1Descriptors, part2Descriptors, matchesList, 2);

        List<DMatch> positiveMatches = new ArrayList<>();
        for (MatOfDMatch matOfDMatch : matchesList) {
            List<DMatch> matchesListA = matOfDMatch.toList();
            for (int i = 0; i < matchesListA.size() - 1; i += 2) {
                DMatch leftMatch = matchesListA.get(i);
                DMatch rightMatch = matchesListA.get(i + 1);
                if (leftMatch.distance < 0.7 * rightMatch.distance) {
                    positiveMatches.add(leftMatch);
                }
            }
        }

        List<DMatch> filteredMatches = new ArrayList<>();

        List<KeyPoint> keypoints1List = keypoints1.toList();
        List<Point> points1List = new ArrayList<>();
        for (KeyPoint kp : keypoints1List) {
            points1List.add(kp.pt);
        }
        MatOfPoint2f points1 = new MatOfPoint2f();
        points1.fromList(points1List);

        List<KeyPoint> keypoints2List = keypoints2.toList();
        List<Point> points2List = new ArrayList<>();
        for (KeyPoint kp : keypoints2List) {
            points2List.add(kp.pt);
        }
        MatOfPoint2f points2 = new MatOfPoint2f();
        points2.fromList(points2List);

        for (DMatch match : positiveMatches) {

            Point pt1 = keypoints1List.get(match.queryIdx).pt;
            Point pt2 = keypoints2List.get(match.trainIdx).pt;

            double distance = Math.sqrt(Math.pow(pt1.x - pt2.x, 2) + Math.pow(pt1.y - pt2.y, 2));

            if (distance >= 677) {
                filteredMatches.add(match);
            }
        }



        MatOfDMatch filteredMatchesMat = new MatOfDMatch();
        filteredMatchesMat.fromList(filteredMatches);
        Mat outputImageA = new Mat();
        Features2d.drawMatches(part1, keypoints1, part2, keypoints2, filteredMatchesMat, outputImageA);




        List<Point> filteredPoints1List = new ArrayList<>();
        List<Point> filteredPoints2List = new ArrayList<>();

        for (DMatch match : filteredMatches) {
            Point pts1 = keypoints1List.get(match.queryIdx).pt;
            Point pts2 = keypoints2List.get(match.trainIdx).pt;
            filteredPoints1List.add(pts1);
            filteredPoints2List.add(pts2);
        }

        MatOfPoint2f filteredPoints1 = new MatOfPoint2f();
        filteredPoints1.fromList(filteredPoints1List);

        MatOfPoint2f filteredPoints2 = new MatOfPoint2f();
        filteredPoints2.fromList(filteredPoints2List);

        //Mat outputImage3 = new Mat();
        //Features2d.drawMatches(part1, keypoints1, part2, keypoints2, filteredMatchesMat, outputImage3);

        //HighGui.imshow("Matches", outputImage3);


        Mat H = Calib3d.findHomography(filteredPoints2, filteredPoints1, Calib3d.RANSAC, 5.0);

        Mat transformedImage = new Mat();
        Imgproc.warpPerspective(part2, transformedImage, H, new Size(part1.cols() + part2.cols(), part1.rows()));

        Mat resultImage = new Mat(transformedImage.size(), transformedImage.type());

        transformedImage.copyTo(resultImage);

        Rect roi = new Rect(0, 0, part1.cols(), part1.rows());

        part1.copyTo(resultImage.submat(roi));


        int blackStripeStartCol = 0;
        for (int col = resultImage.cols() - 1; col >= 0; col--) {
            Mat colMat = resultImage.col(col);
            Scalar sumScalar = Core.sumElems(colMat);
            double sum = sumScalar.val[0] + sumScalar.val[1] + sumScalar.val[2];
            if (sum == 0) {
                blackStripeStartCol = col;
            } else {
                break;
            }
        }

        if (blackStripeStartCol < resultImage.cols() - 1) {
            resultImage = resultImage.colRange(0, blackStripeStartCol + 1);
        }

        HighGui.imshow("result", resultImage);
        HighGui.waitKey(0);

    }
}