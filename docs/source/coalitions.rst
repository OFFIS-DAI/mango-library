==========
Coalitions
==========

Building a coalition in a multi agent framework is often one of the first steps your agents have to execute before they can start to solve the desired problem. We consider coalitions to consist of a connected set of agents, which decided to work together for solving a specific problem. 

As this is a problem, which often have to be solved in multi agent systems, we offer some basic implementations for starting and working withing a coalition.


Building a Coalition
====================

To use the coalition models all of your agents involved in this must have the role `CoalitionParticipantRole`. This role will answer coalition building requests and store so called CoalitionAssignments. A CoalitionAssignment is created when a coalition is confirmed by everybody. To start a coalition you can assign the role CoalitionInitiatorRole with some arguments about the participants of the working circle. A coalition building could look like:

.. code-block:: python3

    c = await Container.factory(addr=('127.0.0.2', 5555))

    # create agents
    agents = []
    addrs = []
    for i in range(10):
        a = RoleAgent(c)
        a.add_role(CoalitionParticipantRole())
        agents.append(a)
        addrs.append((c.addr, a._aid))

    agents[0].add_role(CoalitionInitiatorRole(addrs, 'some application topic', 'application-negotiation'))
    
    # wait until inboxes are empty
    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)
