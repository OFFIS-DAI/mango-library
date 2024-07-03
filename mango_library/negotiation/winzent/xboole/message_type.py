class MessageType:
    Null, EchoRequest, EchoReply, OnlineNotification, OfflineNotification, \
    DemandNotification, OfferNotification, AcceptanceNotification, \
    AcceptanceAcknowledgementNotification, WithdrawalNotification \
        = range(0, 10)
