============
Negotiations
============

A negotiation is a distributed process executed by multiple agents with goal to get a solution for a problem. In multi-agents systems negotiations will happen very frequently so it makes sense to provide a common way of implementing and using them.


====================================
Building upon the negotiations roles
====================================

Unlike the coalition module, the negotiation roles are not usable on their own, because you are missing the actual negotiation logic. As a result a negotiation is abstractly implemented. The core is built with two different roles `NegotiationStarterRole` and `NegotiationParticipant`. The first is responsible for starting a negotiation using an existing coalition and a message creator, which defines the type of negotiations. The second is an extendable role for handling negotiation related messages.

What do you have to do to implement your own negotiation?

#. Create message classes for the negotiation
#. Define a function, which creates a message to start a negotiation
#. Extend NegotiationParticipant and implement handle. Here you can implement your negotiation logic = what should happen when a negotiation related message arrives?

An example for an implemented negotiation can be found in :ref:`COHDA <cohda>`: