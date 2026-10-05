# Lecture 3: Distributed Systems

## CAP theorem

The CAP theorem states that during a network partition a distributed data store
must choose between consistency (every read sees the latest write) and
availability (every request gets a response). Partition tolerance is not optional
in real networks, so the practical choice is CP or AP behaviour under partition.

## Eventual consistency

Eventual consistency guarantees that if no new updates are made, all replicas
converge to the same value. Amazon DynamoDB offers eventually consistent reads by
default and strongly consistent reads as an option at higher cost.

## Load balancing

A load balancer distributes incoming requests across a pool of servers. Common
algorithms are round robin, least connections and consistent hashing. Health
checks remove failing instances from the pool automatically.

## Horizontal vs vertical scaling

Vertical scaling adds resources (CPU, memory) to a single machine and is limited
by hardware ceilings. Horizontal scaling adds more machines behind a load
balancer; it requires stateless services or shared state, but scales further and
improves fault tolerance.
