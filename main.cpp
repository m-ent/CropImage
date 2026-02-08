#include <opencv2/opencv.hpp>
#include <algorithm>
#include <unistd.h>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#define HI  95  // jpeg high quality
#define MID 75  // jpeg middle quality: 60 から変更
#define LOW 25  // jpeg low quality
#define PNG_Q 9
#define GAMMA   // 画像の明るさ調整をガンマ関数で行う

//#define draw_page  // 画像で確認する場合は有効に、普段は無効
//#define first_contour  // 最初の輪郭の確認用、普段は無効
//#define second_contour // 2番目の輪郭の確認用、普段は無効

int show_img(cv::Mat img, const char *title) {
    cv::Mat img_scale;

    cv::resize(img, img_scale, cv::Size(), 0.25, 0.25); // 表示用に縮小
    cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
    cv::imshow(title, img_scale);
    cv::waitKey(0);
    cv::destroyWindow(title);

    return 0;
}

std::vector<cv::Point> get_outer_contour(cv::Mat img) { // 外周を返す
    int threshold = 20; // 2値化の閾値: 暫定値 15
    cv::Mat img_binary;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    int img_width = img.cols, img_height = img.rows;     // img の横, 縦
 
    #ifdef draw_page
        show_img(img, "title");
    #endif

    cv::threshold(img, img_binary, threshold, 255, cv::THRESH_BINARY); // 2値化
    img_binary = ~img_binary;                            // 白黒反転
    cv::findContours(img_binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    int max_area = 0, max_element = 0, tmp_area;
    for(int i = 0; i < contours.size(); i++){  // 最も面積の大きい輪郭を選ぶ (おそらく外周と思われる)
        int x_max = 0, x_min = 65535, y_max = 0;
        for (int j = 0; j < contours[i].size(); j++) {
            if (   (contours[i][j].x > 0.975 * img_width)  // 輪郭に含まれる点が img の左右下 
                || (contours[i][j].x < 0.025 * img_width)  // 2.5%より中央の点だけを扱う
                || (contours[i][j].y > 0.975 * img_height)) {break;};  // 外れる場合は除外
                //|| (contours[i][j].y > 0.75 * img_height)) {break;};  // 外れる場合は除外
            if (contours[i][j].x > x_max) {
                x_max = contours[i][j].x;
            }
            if (contours[i][j].x < x_min) {
                x_min = contours[i][j].x;
            }
            if (contours[i][j].y > y_max) {
                y_max = contours[i][j].y;
            }
        }
        tmp_area = (x_max - x_min) * y_max;  // 輪郭ごとに面積を計算
        if (tmp_area > max_area) {           // 面積のより大きい輪郭を探す
            max_area = tmp_area;
            max_element = i;
        }
    }
    return contours[max_element];       // 最も面積の大きい輪郭を外周とする
}

cv::Mat draw_area_rect(cv::Mat img, cv::RotatedRect rect) { // 外接矩形の描画(by Gemini)
    cv::Mat img_dup = img;
    cv::Point2f pts[4];
    rect.points(pts);

    // Convert the array of float points (Point2f) to a vector of integer points (Point)
    std::vector<cv::Point> polyline_points;
    for (int i = 0; i < 4; i++) {
    // Cast to integer coordinates
        polyline_points.push_back(cv::Point(static_cast<int>(pts[i].x), static_cast<int>(pts[i].y)));
    }

    // Prepare the required vector of vector structure
    const cv::Point* ppt[1] = { &polyline_points[0] };
    int npt[] = { (int)polyline_points.size() };

    // Arguments: image, array of point arrays, number of points per array, 
    //            number of arrays, isClosed, color, thickness, lineType
    cv::polylines(img_dup, ppt, npt, 1, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA); // 破壊的描画
    return img_dup;
}

std::string output_filename(char *input_filename, bool png_flag) {
    int pos, len;
    std::string filename(input_filename);  // 最も一般的な方法

    if ((pos = filename.find(".png")) != std::string::npos) {
        len = 4;
    } else if ((pos = filename.find(".jpg")) != std::string::npos) {
        len = 4;
    } else if ((pos = filename.find(".jpeg")) != std::string::npos) {
        len = 5;
    }
    return filename.replace(pos, len, (png_flag ? ".crop.png" : ".crop.jpg"));
}

int mean_pixel_value(cv::Mat img, int x, int y, int w, int h) {
    int sum = 0, rows = img.rows, cols = img.cols;
    for (int i=0; i < cols; i++) {
        for (int j=0; j < rows; j++) {
            sum +=  (int)img.at<uchar>(i, j);
        }
    }
    return (sum / rows / cols);
}

void usage(char *program_name) {
    std::cout << "Usage: " << program_name << " (option) <inputfile.[png|jpg]> : output jpg file" << std::endl;
    std::cout << "       option: -b: output grayscale file" << std::endl;
    std::cout << "               -p: output png format" << std::endl;
    std::cout << "               -h: high   qualty jpeg: " << HI  << std::endl;
    std::cout << "               -m: middle qualty jpeg: " << MID << std::endl;
    std::cout << "               -l: low    qualty jpeg: " << LOW << std::endl;
}

cv::Mat diff_g2r(cv::Mat img) { // 緑色と赤色の差分を強調
    std::vector<cv::Mat> planes;
    cv::Mat img_blur, result;
    cv::Size ksize = cv::Size(123, 123); // ここのパラメータは適宜(奇数でなければならない)

    cv::GaussianBlur(img, img_blur, ksize, 0); // 差分を取る前にかなりぼかしておく
    img = img_blur;
    cv::split(img, planes); // rgb -> b: planes[0], g: planes[1], r: planes[2]
    result = planes[1] - planes[2];
    return (result * 100);
}

int main(int argc, char *argv[]) {
    cv::Mat img_org, img_diff, img_rotated, img_scale;
    std::vector<cv::Point> outer_contour; // 外周輪郭保存用
    bool bw_flag = false, png_flag = false;  // grascale 出力 flag (default: color)
                                             // png 形式での出力 flag (default: jpg)
    int jpg_quality_bit = 0;  // jpeg quality flag h-m-l
    char *input_file; 
    float side_margin = 1, top_margin = 3, bottom_margin = 4; // 枠外の影の割合の目安
    int option;

    while ((option = getopt(argc, argv, "bphml:")) != -1) { // command line option の解析
        //コマンドライン引数のオプションがなくなるまで繰り返す
        switch (option) {
            case 'b':
                bw_flag = true;       //   grayscale flag: true
                break;
            case 'p':
                png_flag = true;      //   png flag: true
                break;
            case 'h':
                jpg_quality_bit += 4;  //  set jpg_quality_bit h => on
                break;
            case 'm':
                jpg_quality_bit += 2;  //  set jpg_quality_bit m => on
                break;
            case 'l':
                jpg_quality_bit += 1;  //  set jpg_quality_bit l => on
                break;
            default: /* '?' */
                //指定していないオプションが渡された場合
                usage(argv[0]);
                return 1;
                break;
        }
    }
    if (jpg_quality_bit > 4 || jpg_quality_bit == 3) {
        usage(argv[0]);
        return 1;
    }
    if (png_flag && jpg_quality_bit > 0) {
	      std::cout << "png format: option -h/-m/-l is ignored." << std::endl;
    }
    input_file = argv[argc - 1]; //   ファイル名は最後の引数
    img_org = cv::imread(input_file, -1); // 第1引数の画像をopen
    if(img_org.empty()) { return -1;}

    img_diff = diff_g2r(img_org);                          // 緑色と赤色の差分を取って強調
    outer_contour = get_outer_contour(img_diff);           // 外周輪郭を確認する
    cv::RotatedRect rect = cv::minAreaRect(outer_contour); // 輪郭の外接矩形を得る

    // 最初の輪郭の確認用
    #ifdef first_contour
        std::vector<std::vector<cv::Point>> new_contours2;

        new_contours2.push_back(outer_contour);
        cv::drawContours(img_org, new_contours2, 0, cv::Scalar(255, 255, 0));
        show_img(img_scale, "1st recognition");
        return 0;
    #endif

    float angle = rect.angle;                              // 外接矩形の傾き
    float r_angle = (angle > 45.0) ? angle-90.0: angle;    // angle の値によって調整

    cv::Mat M = cv::getRotationMatrix2D(rect.center, r_angle, 1.0);;          // 回転行列
    cv::warpAffine(img_org, img_rotated, M, img_org.size(), cv::INTER_CUBIC); // 回転して img_rotated に保存
                                                                              
    img_diff = diff_g2r(img_rotated); // 緑色と赤色の差分を強調
    outer_contour = get_outer_contour(img_diff);            // 回転後にもう一度外周輪郭を確認する
    rect = cv::minAreaRect(outer_contour);                  // 輪郭の外接矩形を得る

    // 2回目の輪郭の確認用
    #ifdef second_contour
        std::vector<std::vector<cv::Point>> new_contours2;

        new_contours2.push_back(outer_contour);
        cv::drawContours(img_rotated, new_contours2, 0, cv::Scalar(255, 255, 0));
        show_img(img_scale, "2nd recognition");
        return 0;
    #endif

    cv::Point2f vertices[4];                                // 外接矩形の各点を得る
    rect.points(vertices);
    int x_low = 65535, x_high = 0, y_high = 0;
    for (int i = 0; i < 4; i++) {                           // 各点の座標を確認
        if (vertices[i].x < x_low ) {x_low  = vertices[i].x;}
        if (vertices[i].x > x_high) {x_high = vertices[i].x;}
        if (vertices[i].y > y_high) {y_high = vertices[i].y;}
    }

    int width = x_high - x_low;       // ImageMagick の書式に(WxH+X+Y)
    int height = y_high - top_margin;
    std::cout << input_file << ": " << width << "x" << height << "+" << x_low << "+" << top_margin << std::endl;

    #ifdef draw_page
        // ↓ 輪郭表示
        std::vector<std::vector<cv::Point>> new_contours;
        new_contours.push_back(outer_contour);
        cv::Mat img_w_contours = img_rotated;
        cv::drawContours(img_w_contours, new_contours, 0, cv::Scalar(255, 0, 0));
        // ↑ ここまで
    #endif

    std::string output_file = output_filename(input_file, png_flag);  // 出力ファイル名の生成

    /*
    std::cout << "x_low, width, height: " << x_low << "," << width << "," << height <<  std::endl;
    std::cout << "side_margin: " << side_margin <<  std::endl;
    std::cout << "top_margin: " << top_margin <<  std::endl;
    std::cout << "bottom_margin: " << bottom_margin <<  std::endl;
    */

    cv::Mat img_cropped =  cv::Mat(img_rotated, cv::Rect(x_low + side_margin, top_margin, width - side_margin * 2, height - bottom_margin));	// 出力部分の切り出し	

    #ifndef GAMMA
    // 小説などで余白が白い場合、画素値を線形に調整する
        int cropped_width = img_cropped.cols, cropped_height = img_cropped.rows;
        int mean_val[4];
        int sample_points[4][2] = {                                // 上下左右の 1000px x 10px 領域をサンプリング
            {cropped_width / 2,  15},
            {cropped_width - 15,  cropped_height / 2},
            {cropped_width / 2,  cropped_height - 15},
            {15,                 cropped_height / 2}
        };
        for (int i = 0; i < 4; i++) {
             int x = sample_points[i][0], y = sample_points[i][1];
             mean_val[i] = mean_pixel_value(img_cropped, x - 5, y - 5, 10, 10);
        };
        std::sort(mean_val, mean_val + 4);                         // ソートして

        //float gain = 250 / (float)(mean_val[1] + mean_val[2]) * 2; // 最大値と最小値を除いた平均と 250 を比較して
        float gain = 250 / (float)(mean_val[0]);    // 余白の最大値と 250 を比較して
        cv::Mat img_adjusted = gain * img_cropped;	               // 係数 gain を画像に乗算する
    #endif

    #ifdef GAMMA
    // 漫画などで余白が白いとは限らない場合、画素値をgamma 補正で調整する
    // gamma(x) = (log(255) - log(250)) / (log(255) - log(x));
        // 最も白い部分と 250 を比較してgamma値を算出。1行下は実際の計算値
    // float gamma = 0.01980 / (5.54126 - std::log(mean_val[0])); // gamma 値の算出

        /*
        // ヒストグラムを生成するために必要なデータ
        int image_num = 1;      // 入力画像の枚数
        int channels[] = { 0 }; // cv::Matの何番目のチャネルを使うか 今回は白黒画像なので0番目のチャネル以外選択肢なし
        cv::MatND hist;         // ここにヒストグラムが出力される
        int dim_num = 1;        // ヒストグラムの次元数
        int bin_num = 128;       // ヒストグラムのビンの数
        int bin_nums[] = { bin_num };      // 今回は1次元のヒストグラムを作るので要素数は一つ
        float range[] = { 0, 256 };        // 扱うデータの最小値、最大値　今回は輝度データなので値域は[0, 255]
        const float *ranges[] = { range }; // 今回は1次元のヒストグラムを作るので要素数は一つ

        // 白黒画像から輝度のヒストグラムデータ（＝各binごとの出現回数をカウントしたもの）を生成
        cv::calcHist(&img_cropped, image_num, channels, cv::Mat(), hist, dim_num, bin_nums, ranges);

        // テキスト形式でヒストグラムデータを確認
        std::cout << hist << std::endl;
        */

        /*
        cv::Point min_pt, max_pt;
        double minVal, maxVal;
        cv::Mat img_singlechannel = img_cropped;
        cvtColor(img_singlechannel, img_singlechannel, cv::COLOR_RGB2GRAY); // grayscale に(破壊的)

        std::cout << img_singlechannel << std::endl;

        cv::minMaxLoc(img_singlechannel, &minVal, &maxVal, &min_pt, &max_pt);
        //maxVal = 250.0;
        float gamma = 0.01980 / (5.54126 - std::log(maxVal - 5)); // gamma 値の算出

        std::cout << maxVal << "/" << gamma << std::endl;
        */

        float gamma = 1.2; // gamma 値を決め打ちにしてみる

        cv::Mat lut = cv::Mat(1, 256, CV_8U);             // cv::LUT look up table の用意
        for (int i = 0; i < 256; i++) {
                lut.at<uchar>(i) = (uchar)(pow((double)i / 255.0, 1.0 / gamma) * 255.0);
            }
        cv::Mat img_adjusted;	                            // 出力用の画像matrix
        cv::LUT(img_cropped, lut, img_adjusted);	        // gamma補正の適用
    #endif

    if (bw_flag == true) {
        cvtColor(img_adjusted, img_adjusted, cv::COLOR_RGB2GRAY); // bw_flag: true なら grayscale に(破壊的)
    };
    if (png_flag) {
        imwrite(output_file, img_adjusted, {cv::IMWRITE_PNG_COMPRESSION, PNG_Q}); // 出力部分の書き出し
    } 
    else {
        std::vector<int> compression_params;
        int quality;
        switch(jpg_quality_bit) {
	          case 4:
                quality = HI; break;
            case 2:
                quality = MID; break;
            default:
                quality = LOW; break;
        }
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(quality); // JPEGの品質を設定
        imwrite(output_file, img_adjusted, compression_params); // 出力部分の書き出し
    };

    #ifdef draw_page
        std::vector<cv::Point> crop_points;        // 切り抜き用の座標
        crop_points.push_back(cv::Point(x_low,  top_margin));
        crop_points.push_back(cv::Point(x_low,  y_high));
        crop_points.push_back(cv::Point(x_high, y_high));
        crop_points.push_back(cv::Point(x_high, top_margin));

        // ↓ 切り抜き枠表示
        cv::Mat img_dup = img_rotated;
        // Prepare the required vector of vector structure
        const cv::Point* ppt[1] = { &crop_points[0] };
        int npt[] = { (int)crop_points.size() };
        // Arguments: image, array of point arrays, number of points per array, 
        //            number of arrays, isClosed, color, thickness, lineType
        cv::polylines(img_dup, ppt, npt, 1, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA); // 破壊的描画
        // ↑ ここまで

        cv::resize(img_dup, img_scale, cv::Size(), 0.25, 0.25); // 表示用に縮小
        //cv::resize(img_cropped, img_scale, cv::Size(), 0.5, 0.5); // 表示用に縮小
        //cv::resize(img_org, img_scale, cv::Size(), 0.25, 0.25); // 表示用に縮小
   
        cv::namedWindow("Example", cv::WINDOW_AUTOSIZE);
        cv::imshow("Example", img_scale);
        cv::waitKey(0);
        cv::destroyWindow("Example");
    #endif

    return 0;
}

