import os
import sys

def main():
    print("Setting up ANVAYA Demo...")
    
    # Check if models exist
    models_exist = (
        os.path.exists("ml/regime/regime_model.joblib") and
        os.path.exists("ml/probability/heavy_rain_model.joblib") and
        os.path.exists("ml/correction/models")
    )
    
    if not models_exist or "--force" in sys.argv:
        print("Models not found or --force specified. Running training pipeline...")
        # Add root dir to path
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from scripts.train_all import run_training_pipeline
        run_training_pipeline()
    else:
        print("Models already exist. Skipping training. Use --force to retrain.")
        
    print("Demo setup complete. You can now start the backend and frontend.")

if __name__ == "__main__":
    main()
