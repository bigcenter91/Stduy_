import tensorflow as tf

# tf.compat.v1.disable_eager_execution() # 즉시실행모드 안해 = 1.0대 버전
# tf.compat.v1.enable_eager_execution() # 즉시실행모드 해 = 2.0대 버전

print("텐서플로 버전 :", tf.__version__)
print("즉시실행 모드 :", tf.executing_eagerly())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(gpus[0])
    except RuntimeError as e:
        print(e)
else :
    print("GPU 없다!!!")
    

