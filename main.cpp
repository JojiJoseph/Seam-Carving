#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char **argv)
{

    cv::Mat img = cv::imread("Broadway_tower_edit.jpg");
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::imshow("Original", img);
    cv::waitKey(0);

    int W = img_gray.rows;
    for (int iteration = 0; iteration < W / 2; iteration++)
    {
        long long start_time = cv::getTickCount();
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        cv::Mat img_sobelx;
        cv::Mat img_sobely;
        cv::Sobel(img_gray, img_sobelx, CV_64F, 1, 0, 3);
        cv::Sobel(img_gray, img_sobely, CV_64F, 0, 1, 3);
        cv::Mat img_magnitude = img_sobelx.mul(img_sobelx) + img_sobely.mul(img_sobely);
        cv::Mat img_magnitude_normalized;
        cv::normalize(img_magnitude, img_magnitude_normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imshow("Magnitude", img_magnitude_normalized);

        cv::Mat dp = cv::Mat::zeros(img_magnitude.rows, img_magnitude.cols, CV_64F);
        for (int i = img_magnitude.rows - 1; i >= 0; i--)
        {
            for (int j = 0; j < img_magnitude.cols; j++)
            {
                if (i == img_magnitude.rows - 1)
                {
                    dp.at<double>(i, j) = img_magnitude.at<double>(i, j);
                }
                else
                {
                    if (j == 0)
                    {
                        dp.at<double>(i, j) = img_magnitude.at<double>(i, j) + min(dp.at<double>(i + 1, j), dp.at<double>(i + 1, j + 1));
                    }
                    else if (j == img_magnitude.cols - 1)
                    {
                        dp.at<double>(i, j) = img_magnitude.at<double>(i, j) + min(dp.at<double>(i + 1, j - 1), dp.at<double>(i + 1, j));
                    }
                    else
                    {
                        dp.at<double>(i, j) = img_magnitude.at<double>(i, j) + min(dp.at<double>(i + 1, j - 1), min(dp.at<double>(i + 1, j), dp.at<double>(i + 1, j + 1)));
                    }
                }
            }
        }

        int min_index = 0;
        double min_val = dp.at<double>(0, 0);
        for (int j = 0; j < img_magnitude.cols; j++)
        {
            double val = dp.at<double>(0, j);
            if (val < min_val)
            {
                min_val = val;
                min_index = j;
            }
        }

        vector<int> carved_indices;
        auto fp = [&dp, &img_magnitude](int i, int j)
        {
            if (i >= img_magnitude.rows)
            {
                return std::numeric_limits<double>::infinity();
            }
            if (j >= img_magnitude.cols)
            {
                return std::numeric_limits<double>::infinity();
            }
            if (j < 0)
            {
                return std::numeric_limits<double>::infinity();
            }
            return dp.at<double>(i, j);
        };
        cv::Mat img2 = img.clone();

        for (int i = 0; i < img.rows - 1; i++)
        {
            carved_indices.push_back(min_index);

            for (int j = min_index; j < img2.cols - 1; j++)
            {
                img.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(i, j + 1);
            }
            img2.at<cv::Vec3b>(i, min_index) = cv::Vec3b(255, 0, 255);
            float arr[] = {fp(i + 1, min_index - 1), fp(i + 1, min_index), fp(i + 1, min_index + 1)};
            int delta = std::min_element(begin(arr), end(arr)) - begin(arr) - 1;
            min_index = min_index + delta;
        }
        cv::imshow("Line to carve", img2);

        img = img(cv::Rect(0, 0, img.cols - 1, img.rows));
        long long end_time = cv::getTickCount();

        cout << "Time taken for 1 iteration: " << (end_time - start_time) / cv::getTickFrequency() * 1000 << "ms" << endl;
        cv::imshow("Carved", img);
        char key = cv::waitKey(1) & 0xFF;

        if (key == 'q')
            break;
    }
    cv::imshow("Final", img);
    cv::waitKey(0);

    return 0;
}
