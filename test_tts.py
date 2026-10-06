import pyttsx3, threading, time
def t():
    print('init')
    try:
        e=pyttsx3.init()
        print('say')
        e.say('test')
        print('run')
        e.runAndWait()
        print('done')
    except Exception as ex:
        print('error:', ex)
threading.Thread(target=t).start()
time.sleep(5)
