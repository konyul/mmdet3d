import smtplib
from email.mime.text import MIMEText
import time
start_time = time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(time.time()+32400))

# 세션 생성
s = smtplib.SMTP('smtp.gmail.com', 587)

# TLS 보안 시작
s.starttls()

senderAddr = 'konyulpark@gmail.com'
recipientAddr = "pak199498@naver.com"

# 로그인 인증

s.login(senderAddr, 'mhqu ojzt didk lmme')

text="training is over"

msg=MIMEText(text)
msg['Subject']=start_time+'done'
msg['From']=senderAddr
msg['To']=recipientAddr

s.sendmail(senderAddr, [recipientAddr], msg.as_string())
s.quit()