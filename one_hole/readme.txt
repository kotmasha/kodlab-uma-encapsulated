The goal of this simulation is for us to figure out correct communication between agents through the experiment.

Here, the experiment describes TWO agents, ALICE and BOB, controlling the SAME "body" traveling through a once punctured
gridlike environment.

Both ALICE and BOB have access to the same actions, motion in either one of the grid directions, "up"/"dn"/"rt"/"lt".

However, if BOB happens to "vote" against ALICE (e.g. ALICE votes "up" while BOB votes "dn"), the two votes will cancel 
each other out.

ALICE possesses the basic "GPS" sensors for the rectangular grid, allowing her to learn the global grid structure (in its 
poc-dual form) and to plan motion towards a target using the propagation method.

BOB, however, possesses an awareness of the puncture, and is expected to eventually "vote ALICE down" each time she tries
to bump into the puncture. 

We will test out different strategies BOB could follow in order to achieve this goal.
