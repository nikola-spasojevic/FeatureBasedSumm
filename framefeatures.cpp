#include "framefeatures.h"

FrameFeatures::FrameFeatures(QObject *parent) :
    QThread(parent)
{

}

void FrameFeatures::run()
{
    found = false;
    dictionaryCreated = false;
    processFrames();
}

void FrameFeatures::processFrames()
{
    VideoCapture *capture  =  new cv::VideoCapture(filename);
    int numberOfFrames = capture->get(CV_CAP_PROP_FRAME_COUNT);
    int frameRate = (int) capture->get(CV_CAP_PROP_FPS);
    int Nth = frameRate;
    int j = 0;
    Mat frm;
    capture->read(frm); // get a new frame from camera

    vector<KeyPoint>  keypoints_scene;
    vector<vector<cv::Point> > mser_regions;
    Mat descriptors_scene;

    Ptr<FeatureDetector> detector =  new PyramidAdaptedFeatureDetector( new DynamicAdaptedFeatureDetector ( new SurfAdjuster(700,true), 500, 1000, 3), 4);
    //Ptr<FeatureDetector> detector = FeatureDetector::create("MSER");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");

        //SIFT
        //Ptr<FeatureDetector> detector = new cv::SiftFeatureDetector();
        //Ptr<DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();

        //ORB
        //cv::FeatureDetector* featureDetector = new OrbFeatureDetector();
        //cv::DescriptorExtractor* descriptorExtractor = new OrbDescriptorExtractor();

        //FAST + SIFT
        //cv::FeatureDetector* featureDetector = new FastFeatureDetector();
        //cv::DescriptorExtractor* descriptorExtractor = new cv::SiftDescriptorExtractor();

        //MSER + SIFT
        //Ptr<FeatureDetector> detector = new cv::MserFeatureDetector();
        MSER ms;
        //Ptr<DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();

        //MSER + FREAK
        //Ptr<FeatureDetector> detector = new cv::MserFeatureDetector();
       // Ptr<DescriptorExtractor> extractor = new cv::FREAK;

        //SURF
        //Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
        //Ptr<FeatureDetector> detector = new DynamicAdaptedFeatureDetector ( new SurfAdjuster(700,true), 10, 500, 10);

        //Ptr<FeatureDetector> detector = FeatureDetector::create("HARRIS");
        //Ptr<DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();

        //SURF + FREAK
        //cv::FeatureDetector* featureDetector = new cv::SurfFeatureDetector();
        //cv::DescriptorExtractor *descriptorExtractor = new cv::FREAK();

        //KAZE
        //cv::FeatureDetector* featureDetector = new cv::KazeFeatureDetector();
        //cv::DescriptorExtractor* descriptorExtractor = new cv::KazeDescriptorExtractor();


    while( j < (numberOfFrames-Nth+1) && !frm.empty() )
    {
        keypoints_scene.clear();
        cv::Mat mask(0, 0, CV_8UC1);
        //cvtColor(frm, frm, CV_BGR2GRAY);

        //ms(frm, mser_regions);  //<=================================MSER

        for(int i = 0; i < mser_regions.size(); ++i)
        {
            size_t count = mser_regions.at(i).size();
            if( count < 6 )
                continue;

            Mat pointsf;
            Mat(mser_regions.at(i)).convertTo(pointsf, CV_32F);
            RotatedRect box = fitEllipse(pointsf);

            if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30 )
                    continue;

            ellipse(frm, box, Scalar(0,0,255), 1, CV_AA);
            //ellipse(frame, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(0,255,255), 1, CV_AA);
        }





        //-- Step 1: Detect the keypoints using Detector
        detector->detect(frm, keypoints_scene, mask);

        for (int k = 0; k < keypoints_scene.size(); k++)
        {
            float angle = keypoints_scene.at(k).angle;
            //Size size = Size(keypoints_scene.at(k).size, );
            Point point = keypoints_scene.at(k).pt;
            //ellipse(frm, point, size, angle, 0, 360, Scalar(0,255,255), 1, CV_AA);
        }

        //-- Step 2: Calculate descriptors (feature vectors)
        extractor->compute(frm, keypoints_scene, descriptors_scene);

        //-- Passing values to mainwindow.cpp
        frameVector.push_back(frm);
        keypoints_frameVector.push_back(keypoints_scene);
        descriptors_sceneVector.push_back(descriptors_scene);

        qDebug() << keypoints_scene.size() << ",";
        //qDebug() << "Size of Scene Descriptor:  " << descriptors_sceneVector.size() << ": " << descriptors_scene.rows << " x " << descriptors_scene.cols;

        if (!descriptors_scene.empty())
        {
            bowTrainer->add(descriptors_scene);
        }

        j += Nth;
        capture->set(CV_CAP_PROP_POS_FRAMES, j);
        capture->read(frm);// get every Nth frame (every second of video)
        found = true;
    }

    calculateCluster();

    emit onFeaturesFound(found);
    emit onDictionaryMade(dictionaryCreated);
}

void FrameFeatures::setFilename(string filename)
{
    this->filename = filename;
}

vector<cv::Mat > FrameFeatures::getFeatureVectors()
{
    return this->descriptors_sceneVector;
}

void FrameFeatures::calculateCluster()
{
    /************* TRAINING VOCABULARY **************/
    //Training the Bag of Words model with the selected feature components
    vector<Mat> descriptors = bowTrainer->getDescriptors();

    int count=0;
    for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
    {
        count+=iter->rows;
    }
    qDebug() << "Clustering " << count << " features" << endl;

    if (count > DICTIONARY_SIZE)
    {
        dictionary = bowTrainer->cluster();

        if (!dictionary.empty())
        {
            dictionaryCreated = true;
        }

        qDebug() << "dictionary size: "<< dictionary.rows << " x " << dictionary.cols;
    }
}

