package facedetect;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;

import scala.Function1;
import scala.Tuple2;
import scala.math.Ordering;
import scala.reflect.ClassTag;

public class FaceDetect {

	public static void main(String[] args) throws IOException {
		System.setProperty("hadoop.home.dir", "D:\\SOFT\\EclipseWorkspace\\hadoop-3.3.0\\");
		//Create Spark config and Spark Context
		SparkConf conf = new SparkConf()
				//Master
				.setMaster("spark://hdfs.ddns.net:7077")
				.setAppName("FaceDetect")
				//Set driver
				.set("spark.driver.host", "192.168.0.105")
				.set("spark.driver.bindAddress", "192.168.0.105")
				.setJars(new String[] {"D:\\SOFT\\EclipseWorkspace\\facedetect\\target\\facedetect-0.0.1-SNAPSHOT.jar"});
		JavaSparkContext sc = new JavaSparkContext(conf);
		//Load OpenCV core
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//Load face detector
		CascadeClassifier faceDetector=new CascadeClassifier("res/haarcascade_profileface.xml");
		//Load people detector
		HOGDescriptor hOG=new HOGDescriptor();
		hOG.setSVMDetector(HOGDescriptor.getDefaultPeopleDetector());
		//Read all images in hdfs
		JavaPairRDD<String, PortableDataStream> rdd=sc.binaryFiles("hdfs://hdfs.ddns.net:8020/humans");
		//Mapping PortableDataStream (image data) to byte[], convert RDD to list of tuple
		List<Tuple2<String, byte[]>> imgs=rdd.map(new Function<Tuple2<String,PortableDataStream>, Tuple2<String,byte[]>>() {

			@Override
			public Tuple2<String, byte[]> call(Tuple2<String, PortableDataStream> t) throws Exception {
				// TODO Auto-generated method stub
				return new Tuple2<String, byte[]>(t._1(), t._2.toArray());
			}
		}).collect();
		//Loop all tuples (image)
		imgs.stream().forEach(tuple->{
			//Convert byte[] of the image to Mat
			Mat mat=Imgcodecs.imdecode(new MatOfByte(tuple._2()), Imgcodecs.IMREAD_UNCHANGED);
			//Convert origin mat (image) to gray version and flip version
			Mat gray=new Mat();
			Mat flip=new Mat();
			Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
			Core.flip(gray, flip, 1);
			//Call face detect func
			detectFace(mat, faceDetector, gray, flip);
			//Call people detect func
			detectPeople(mat, hOG, gray);
			try {
				//Call save image to HDFS func
				saveToHDFS(mat, tuple);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		});
	}
	private static void saveToHDFS(Mat mat, Tuple2<String, byte[]> tuple) throws IOException {
		//Convert Mat (image) to byte[]
		MatOfByte mob=new MatOfByte();
		Imgcodecs.imencode(".jpg", mat, mob);
		byte[] output=mob.toArray();
		//Set HDFS address for saving image
		Configuration config = new Configuration();
		config.set("fs.defaultFS", "hdfs://hdfs.ddns.net:8020");
	    FileSystem fs = FileSystem.get(config);
	    //Create path of the image, this image will be wrote to HDFS
	    String s = fs.getHomeDirectory()+"/"+tuple._1().split(",")[1];
	    Path path = new Path(s);
	    //Create output stream to HDFS
	    FSDataOutputStream out = fs.create(path);
	    //Flush before write
	    out.flush();
	    //Write image to HDFS path
	    out.write(output);
	    //Flush after write
	    out.flush();
	    //Logging
		System.out.println("Process "+tuple._1());
		//Close output stream
	    out.close();
	}
	private static void detectFace(Mat mat, CascadeClassifier faceDetector, Mat gray, Mat flip) {
		//Create a rect, it will be printed on image 
		MatOfRect faceDetections=new MatOfRect();
		//Detect face in gray version
		faceDetector.detectMultiScale(gray, faceDetections, 1.25, 3);
		//Print rect to origin image
		for (Rect rect : faceDetections.toArray())
        {	
            Imgproc.rectangle(mat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255), 2);
            
        }
		//Detect face in flip version
		faceDetector.detectMultiScale(flip, faceDetections, 1.25, 3);
		//Print rect to origin image
		for (Rect rect : faceDetections.toArray())
        {	
            Imgproc.rectangle(mat, new Point(mat.width()-rect.x-rect.width, rect.y), new Point(mat.width()-rect.x-rect.width + rect.width, rect.y + rect.height), new Scalar(0, 0, 255), 2);
        }
	}
	private static void detectPeople(Mat mat, HOGDescriptor hOG, Mat gray) {
		//Create a rect, it will be printed on image
		MatOfRect peopleDetections = new MatOfRect();
		//Detect people in gray version by HOG algorithm (Histogram of oriented gradients: Bieu do gradient co huong)
        hOG.detectMultiScale(gray, peopleDetections, new MatOfDouble(), 0, new Size(4, 4),
                new Size(8, 8), 1.1, 8, false);
        //Print rect to origin image
        for (Rect rect : peopleDetections.toArray())
        {	
            Imgproc.rectangle(mat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 2);
            
        }
	}
}
