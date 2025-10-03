#!/usr/bin/env python3
"""
Debug script to check which parameters were actually trainable during training
"""

import os
import sys
import tensorflow as tf
from myxtts.models.xtts import XTTS
from myxtts.config.config import XTTSConfig

def debug_trainable_parameters():
    """Check which model parameters are trainable"""
    
    print("🔍 Debugging Trainable Parameters...")
    print("=" * 60)
    
    try:
        # Load checkpoint metadata to get config
        checkpoint_dir = "./checkpointsmain"
        metadata_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_metadata.json')]
        
        if not metadata_files:
            print("❌ No checkpoint metadata found")
            return
            
        latest_metadata = sorted(metadata_files)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_metadata.replace('_metadata.json', ''))
        
        print(f"📁 Loading checkpoint: {checkpoint_path}")
        
        # Load the model
        model = XTTS.load_checkpoint(checkpoint_path)
        
        print("\n🧠 Model Architecture Analysis:")
        print("=" * 40)
        
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        # Analyze each component
        components = {
            'text_encoder': getattr(model, 'text_encoder', None),
            'audio_encoder': getattr(model, 'audio_encoder', None), 
            'decoder': getattr(model, 'decoder', None),
            'mel_head': getattr(model, 'mel_head', None),
            'speaker_encoder': getattr(model, 'speaker_encoder', None),
        }
        
        for name, component in components.items():
            if component is None:
                print(f"❌ {name}: Not found")
                continue
                
            comp_total = 0
            comp_trainable = 0
            
            try:
                if hasattr(component, 'trainable_variables'):
                    for var in component.trainable_variables:
                        param_count = tf.size(var).numpy()
                        comp_trainable += param_count
                        trainable_params += param_count
                        
                if hasattr(component, 'non_trainable_variables'):
                    for var in component.non_trainable_variables:
                        param_count = tf.size(var).numpy()
                        non_trainable_params += param_count
                        
                if hasattr(component, 'variables'):
                    for var in component.variables:
                        param_count = tf.size(var).numpy()
                        comp_total += param_count
                        
                total_params += comp_total
                
                print(f"🔧 {name}:")
                print(f"   Total: {comp_total:,} parameters")
                print(f"   Trainable: {comp_trainable:,} parameters")
                print(f"   Trainable: {component.trainable if hasattr(component, 'trainable') else 'Unknown'}")
                
            except Exception as e:
                print(f"❌ Error analyzing {name}: {e}")
        
        print("\n📊 Overall Summary:")
        print("=" * 40)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.1f}%")
        
        # Check model-level trainable setting
        print(f"\n🏗️ Model Level Settings:")
        print(f"model.trainable: {model.trainable if hasattr(model, 'trainable') else 'Unknown'}")
        
        # Check specific critical components
        print(f"\n🎯 Critical Component Status:")
        critical_components = ['decoder', 'mel_head']
        for comp_name in critical_components:
            comp = getattr(model, comp_name, None)
            if comp:
                trainable_status = comp.trainable if hasattr(comp, 'trainable') else 'Unknown'
                print(f"   {comp_name}.trainable: {trainable_status}")
                
                # Check layer-level trainable status
                if hasattr(comp, 'layers'):
                    trainable_layers = sum(1 for layer in comp.layers if getattr(layer, 'trainable', True))
                    total_layers = len(comp.layers)
                    print(f"      Trainable layers: {trainable_layers}/{total_layers}")
            else:
                print(f"   {comp_name}: Not found")
        
        print("\n" + "=" * 60)
        
        # Additional diagnosis
        print("🔍 Possible explanations for loss decrease without learning:")
        print("1. Only auxiliary components (batch norm, etc.) were updating")
        print("2. Loss calculation issues or normalization changes")
        print("3. Data preprocessing changes during training")
        print("4. Optimizer momentum effects without weight updates")
        print("5. Loss clipping/scaling artifacts")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params/total_params if total_params > 0 else 0,
            'model_trainable': getattr(model, 'trainable', None)
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_trainable_parameters()