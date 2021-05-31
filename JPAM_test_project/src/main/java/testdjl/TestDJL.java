package testdjl;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;

/**
 * Test loading a Tensorflow classifier using the jdl4pam library. 
 * @author Jamie Macaulay
 *
 */
public class TestDJL {

	public static void main( String[] args ) {

		if (args==null || args.length<2) {
			args = new String[2];
			args[0] = "C:\\Users\\au671271\\Google Drive\\PAMGuard_dev\\Deep_Learning\\Right_whales_DG\\model_lenet_dropout_input_conv_all\\saved_model.pb";
			//		args[1] = "C:\\Users\\au671271\\Google Drive\\PAMGuard_dev\\tutorials\\deep_learning\\bat_DL_tutorial\\BAT_MULTI_JAMIE_5ms_256fft_10hop_MM_10_90_128_256_JIT_PITCHTIMECHANGE_AUG_V1.pk";
			args[1]= "C:\\Users\\au671271\\Desktop\\bat_pytorch\\Copy of BAT_JAMIE_4ms_256fft_8hop_-100_20_15_60_128_256_NOJIT_BAT_DATA_NAUG_V1_JIT.pk";
		}
		
		String modelPath = args[0]; 
		File file = new File(modelPath);
		Path modelDir = Paths.get(file.getAbsoluteFile().getParent()); 
		String modelName = file.getName(); 

		System.out.println(Engine.getAllEngines()); 

		System.out.println("Loading model 1: " + modelName);


		Model model = Model.newInstance(modelPath, "TensorFlow");

				try {
					model.load(modelDir, modelName);
					System.out.println("Model loaded successfully: " + modelName);
				} catch (MalformedModelException | IOException e) {
					e.printStackTrace();
				}

		modelPath = args[1]; 

		file = new File(modelPath);
		modelDir = Paths.get(file.getAbsoluteFile().getParent()); 
		modelName = file.getName(); 

		System.out.println("Loading model 2: " + modelName);

		System.out.println(Engine.getAllEngines()); 
				model = Model.newInstance(modelPath, "PyTorch");
				try {
					model.load(modelDir, modelName);
					System.out.println("Model loaded successfully: " + modelName);
				} catch (MalformedModelException | IOException e) {
					e.printStackTrace();
				}

	}
}
