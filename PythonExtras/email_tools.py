import smtplib


def send_email(host, port, user, password, toAddress, subject, body):

    fromAddress = user
    payloadBits = [
        'From: {}'.format(fromAddress),
        'To: {}'.format(toAddress),
        'Content-Type: text/html',
        'Subject: {}'.format(subject),
        '',
        body
    ]
    payload = '\r\n'.join(payloadBits)

    with smtplib.SMTP(host, port) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(user, password)
        smtp.sendmail(fromAddress, toAddress, payload)


def send_bot_email_report(subject, body):
    send_email(
        host='mail.demiarch.de',
        port=587,
        user='bot@demiarch.de',
        password='xvjvOzCxR2FVya4cWivU',
        toAddress='architect@demiarch.de',
        subject=subject,
        body=body
    )