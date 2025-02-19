# Meetings

[https://tudelft.zoom.us/j/3491351193?pwd=UXhpVjZNNjdpWm9vcHR0VUFRbGFSUT09#success](https://tudelft.zoom.us/j/3491351193?pwd=UXhpVjZNNjdpWm9vcHR0VUFRbGFSUT09#success)

### Meeting 1: Exploratory meeting (01-03)

- Introduction of me (BSc, Bridging, Dreamteam, MSc, Artemis, Ariston)
- My interests
- Astrodynamics and ML/AI and programming in general (C++ course)
    - Talked to you already about thesis topics in first few weeks
- Courses: all Astrodynamics courses, Prob/stat, SL for Engineers, Machine Learning 1
    - Have all ECTS now
- Kevin’s expertise and possible & worthy thesis topics
- Digging into topics

### Meeting 2: Questions/things to discuss (28-03)

### Agenda meeting 2

- Questions shape based method
- Questions PINNs
- General remarks and idea’s/interest
- Kevin’s idea’s/views on my idea’s
- Other questions

### **PINNs**

- How it $t$ actually inputted?
⇒ By a training batch $\begin{aligned}T & =\left\{\mathcal{G}\left(\mu=t_0+n \Delta t, \sigma=0.2 \Delta t\right) \mid n \in\{0,1, \ldots, M\}\right\} \\\Delta t & =\frac{t_f-t_0}{M}\end{aligned}$ (an equidistant grid of time points is used as the base sample, which is then perturbed randomly for each epoch)
- How is the NN trained? Figure 2 → what $\mathbf{z}, \mathbf{u}, t_j$ is put in in $\mathcal{L}_d=\frac{1}{M} \sum_j^M[\dot{\mathbf{z}}-f(\mathbf{z}, \mathbf{u}, t)]^2$? The $\dot{\mathbf{z}}$ comes from the NN, where does the $\mathbf{z}, \mathbf{u}$ come from?
⇒ Try to think about it again, answers that I came up with myself is something like: you put in the positions into the EOM’s and then you know what the acceleration should be, *what about velocity though..??*
- Every 1000 epochs, a test evaluation of the network is carried out. This comprises the evaluation of 200 training points in an unperturbed equidistant time grid and the outcome is used to evaluate the metrics $d x, d v \text { and } d m$. The state of the network at the epoch with minimum total test loss is considered the solution. Note that this is not necessarily the final epoch → I don’t get this

Remarks:

- I do need to study more how neural networks work → Deep Learning course assignments
*⇒ True, if needed we can start thesis a few weeks later so that you get a head start*
- I feel like when doing a thesis topic on PINNs, a very large part of your thesis will be tuning your PINN and with improving shape based methods with Machine Learning, the work you do is a bit more diverse (and a bit more astrodynamics related)

Idea’s:

- Make it supervised by having optimal solutions from known problems (GTOC?)
    - A strength of PINNs might be the ability to be trained with no labelled data…
    *⇒ Could be, but this depends on what you apply it to though*
- Generalization would probably the most interesting thing to work on, so I think doing research that attributes to that is important.
    - CR3BP → would it really be that much work you think? What changes? → EOMs change…
    ⇒ You really need to get into the CR3BP… which can be a lot of extra work, so he is just hesitant to do this, so maybe not try this. It’s not a definitive NO though, but it is risky he thinks…
    - Ascent and descent trajectory
    *⇒ Is not Kevins main expertise, so maybe do another problem like a fly by/gravity assist.*
    - Perturbations (SRP, perturbing bodies)
- ~~Architecture change → Although many aspects of the configuration can be researched and optimized, this work will only focus on the training procedure and the selection of the loss weights.~~ EDIT: not true, see appendix for other findings of the research

### **ML to improve Hodographic Shaping**

- Check if I understand how the NN is implemented…?
⇒ What I understand is that you use a (trained) NN to estimate the $\Delta V$, instead of calculating $\Delta V$ by numerical integration. This saves a lot of time as no numerical integration has to be done. Num integration can take like 2 whole seconds and the NN does this in a small fraction of a second. The free parameters are still optimized, like in the normal shaping method, by an optimiser method like Nelder-Mead, DE (Differential Evolution), BOBYQA (Bound Optimization BY Quadratic Approximation)
- I don’t see how the inputs in figure 1 are actually put into the NN
⇒ By encoding I think…
- TEST CASES → According to figure 1, a trajectory and its corresponding delta V are already based on a dep date and ToF right... so do you still optimize for this..?
⇒ didn’t fully understand the answer, but something with an inner loop and an outer loop… just think about it a bit more I think
- What exactly is meant by inward/outward transfer? how does inward look like?
⇒ Inward is just to an inner planet (e.g. mars to venus), outward is to an outer planet
- What exactly is meant by rendez-vous here?
⇒ I think only the part of the trajectory between the departure and arrival problem orbit, so e.g. for earth-mars transfer everything from leaving e.g. earth orbit to right before insertion of e.g. mars orbit ← actually in this thesis it means that the trajectory ends up in the orbit around the planet, so no insertion burn or something. In real life this is however a choice you can make.
- Why did Tjomme just choose arbitrary shape functions for the 9-DoF model? 
⇒ There is no work done on this yet, so I assume just no time to research it ← He has thought about it a bit though, but he just didn’t explain it a lot in his thesis, but indeed a shortage of time caused him to not being able to fully research this

Remarks:

Idea’s:

- Most important aspect of this research is to make a generalized model. The ML models are way faster than the traditional hodographic shaping methods, however training time of the NN is not taken into account. If you only have to train a single NN that you can use for all type of transfers, than this is a valid assumption, but as of yet not consistent general NN has been made. The short feature selection that has been done shows that including more features generally improves the model.
⇒ So more research into feature engineering might be the most interesting thing to do here.
- 9 DoF
    - NN architecture tuning
    - New shaping functions investigation
- Incorporation of non-rendezvous trajectories → NN architecture tuning → most work apparently

⇒ You can’t do the first 2 things together in 1 thesis, that would be too much, you have to choose

### **Other questions**

- What is the relevance/potential of both topics for the working field?
⇒ don’t remember the answer…
- Have you talked to other students as well?
⇒ briefly and informally, he has to do that a bit more, since ideally he wants them to start close after me.
- What exactly is your experience in the field. When did you start to get into AI related stuff?
⇒ He started with ML stuff like 7 years ago
- Are there any phd students working on similar topics that I could potentially discuss things with?
⇒ Kevin has no phd students, he has no budget for this, he is only a lecturer
- What library to use for PINNs?
⇒ Go for Scikit Learn, since this is something Kevin has expertise with as well. Furthermore Sklearn is not necesarrily a NN’s focused library, so if during the thesis we decide to derail a bit from the plan, you could still use Sklearn in the thesis, whereas this would not be the case if you had used a library that focusess solely on NN’s.

### **Random thoughts**

- *SPAICE*
- *What about using a PINN in the shaping method?*
- *Apply stuff to GTOC problems? For these problems lot’s of experts have tried to find solutions, so you can really see how it competes with state of the art solutions.*

### Next steps

I’ve let Kevin know that I would probably want to continue with the PINNs, but could still deviate from that and choose for the Shape Based.

For now:

- Come up with an idea for a topic you want to explore yourself, then next meeting we can probably finalize that and decide in detail what my thesis topic can be about

 **NB:** Kevin kind of assertains/guarantees that doing a thesis topic with him is possible, so I can let S. Gehly know I’m not doing something with him anymore.

### Meeting 3: More detailed idea of thesis topic (08-04)

### Agenda meeting 3

- Present idea’s
- PyTorch/Tensorflow
    - GPU training/ advanced NN
- Next steps
    - Document the plan just to have it black on white
        - Preparatory work (~2 weeks)
            - Kick off planning
- Additional questions

**More practical research topics - different scenarios**

- Gravity assists maneuvers / fly-by’s
*These assist maneuvers can help spacecraft gain velocity or redirect to new destinations with minimal propellant use.*
*If you have PINNs capable of optimizing transfers and gravity assist, a new spectrum of trajectories for missions to e.g. asteroids, outer solar system planets, Kuiper belt*
    - Apply PINN to a gravity assist
    - Combine transfers and gravity assist into 1 PINN model (e.g. apply it to a mission to e.g. an asteroid)

⇒ This is the chosen thesis topic

- Interplanetary Missions Beyond Mars, e.g. missions targeting other bodies, such as Venus, Jupiter's moons
*These missions will have different times of flight, maybe more revolutions around the sun. Outward transfers might be similar to Thomas’ thesis, so maybe focussing on inward transfers will provide more research output (we’ll learn more from it).*
*It can be experimented what the difference is in applying a PINN to inward transfers , to different bodies, and varying the amount of revolutions around the sun*
    - Apply PINN to an inward transfer
    - Investigate multiple transfers to different bodies like Jupiter

- Lunar Ascent trajectories
*Lunar ascent trajectories will be relevant in the near future, e.g. because of the Artemis program.*
    - Apply PINN to high trust ascent trajectory

⇒ Not Kevin’s expertise. Also, Kevin noticed that someone else somewhere is already doing this so we should not also do it then

**More Theoretical - PINN architecture - Mathematical research topics**

- When applying the PINN to a different scenario, (parts of) the PINN architecture have to be investigated again too. Insights from Thomas’ thesis can be used to effectively determine what has to be investigated
    - Investigation of fully connected networks vs parallel networks (again)
    The consistency and slightly superior values of $dx$ and $dv$ for the parallel network was prioritized in this work. However, fully-connected neural networks are nearly equally capable of finding solutions to transfer problems and they can do so much more efficiently, making them an attractive option worthwhile for further investigation.
    - Investigate influence of amount of layers/neurons, steepness parameter, training samples, activation functions, learning rate (schedules), loss weights
- Investigate the use of Cartesian coordinates?
    - It didn’t work for transfers, because for cartesian coordinates there is no natural method to embed the amount of revolutions into the constraints $x_f$ and $y_f$. I think they could work for other scenario’s without revolutions?
- Adaptive loss weight scheme
With this you can navigate the loss landscape more efficiently. It also builds towards multi-objective optimization.
    - Come up with an adaptive loss weight scheme
- Try a time objective
See how the PINN handles a different, very relevant, objective
    - Include a trainable parameter (time) by scaling the time input with this trainable parameter representing the open final time parameter.
        - Try multiple objective optimization

⇒ This might be a whole thesis topic on it’s own!

- 3D instead of 2D
Provides more realistic trajectories, e.g. you can then have transfer between two orbits in different orbital planes
    - Implement 1 more coordinate → from polar to spherical coordinates
    - Implement 2 more EOM’s and 1 extra control parameter

*⇒ Also, this might be a whole thesis topic on it’s own!*

- There are multiple ways to make a NN physics informed/constrained. Thomas did this by including the objective as an additional term to the loss function, but there are more ways to do it, might be worth investigating this (during my literature study?). You can also combine things, you can e.g. also initialize the NN’s weights and biases to reflect known physics constraints.

*⇒ Definitely to this during the literature study!*

- Try a supervised implementation of the PINN
*This might provide more optimal solutions as the PINN is now used to fine-tune a initial solution which is already nearing an optimal, maybe faster training times as well*
    - Combine Hodographic-Shaping with PINN
        - Using data points from a solution acquired from hodographic shaping as data input to the PCNN
        - Try having the internal network architecture of the PCNN be derived from the base functions
    - Try to apply the supervised PINN to a difficult mission, e.g. a mission with multiple transfers and fly-by’s

*⇒ This might be even more than a whole thesis*

**Plan until kick-off**

- Learn using Keras!
- *Some* Deep Learning lectures and assignments
- Big Blue Server → Super computer TU Delft. He said maybe become familiar with this, right before that he asked me what my preferred programming IDE was (Pycharm, Spyder etc.)
- Research Methodologies course - see what is useful
- Video structured thinking Kevin
- Look for textbooks on NN in general and PINNs
- Choose 80 FTE or 100 FTE
- Calmcode video’s
- Overleaf tutorial

**Additional questions:**

- PyTorch or Tensorflow?
⇒ Keras.
- Updates on other students?
⇒ nope not yet
- Research Methodologies
⇒ see what’s useful
- How to implements gravity assist?
⇒ Sphere of influence will probably not be entered during a gravity assist, he said something about patched (conics?) … learn about this in FoA maybe?

**Random idea’s:**

- **Integrate with Mission Design Tools**: Integrating PINN models with existing mission design and analysis tools could facilitate the adoption of this novel approach in the aerospace industry, allowing for more complex and efficient mission planning.
- What if we add to the neural network: the positions of multiple bodies/planets, which are dependent on and thus change in time, to try to acquire optimal trajectories including fly-by’s, why would this useful though..?
- Solving CR3BP in a supervised manner, instead of unsupervised, e.g. by incorporating the hodographic shaping solutions to steer the PINN in a certain direction

### Meeting 4: Week -1 (07-05)

### Agenda

- AI generated Image
- Summary past weeks
    - Keras
    - DL lectures + assignments
    - Research methodologies
    - Delft Blue
    - Busy with other stuff…
- 80% FTE
*Kevin agrees totally with this, the reasons I gave were found to be good reasons by Kevin*
- Frequency of Meetings
*Once a week is a good starting point*
- Format of Research plan deliverable
*Don’t exactly recall what was discussed, but basically something in the trend that a report was the desired deliverable*
- Literature study
    - Gravity assist / EOMs
    - NN
    - PINNs state of the art / useful stuff (not necessarily state of the art)
    - Keras/programming
    - Structured thinking video
    - Delft Blue course 18 June
    - Research methodologies
    
    *All seems fine*
    
- Signing thesis Kick-off form

- Other comments Kevin:
    - **During the whole Thesis: You are in the driving seat, so you have to take the lead on things, e.g. if I forget about things please remind me**
    - I would like you to keep a log of what you’ve been doing on a daily basis
    - You can look at thesis of Tommy Kranen for your literature study on gravity assist maneuvers
    - Kevin is 3 weeks on holiday 24-05 - 15-06
    
    - Literature study is for
        - Looking for a research gap
        - Get up to speed with the state of the art research so that you don’t reinvent the wheel. Also get up to speed with not necessarily only the state of the art research but also all useful tools that have been developed over the years. Sometimes that can be even more useful than state of the art things
        - Filling up personal knowledge gaps (I said this out of myself, but he agreed to it)
        - Having a ‘handbook’ for the rest of your thesis. It is the plan of attack for your thesis.

### Meeting 5: Week 0 (16-05)

### Agenda

- AI generated Image
- Logbook
- Findings Readings papers
    - Difficulty understanding papers → mathematical papers
    *I kind of assumed I needed to solve an OCP in my thesis, probably true, but I need to have arguments on why to solve an OCP. In the end what we want is to have an optimum trajectory, probably with solving OCP you get that, but why not heuristic? → need to reason why.*
    - How to search?
    - How to deal with complex papers you don’t understand
- Planning
- Kick-off Form
***Decided to extend the kick off by 2 weeks! (so to May 20)***

### Meeting 6: Week 1 (22-05) (Kick off meeting)

### Agenda

- AI image
- Planning
*If you’re gone for a vacation it could take some time, once you get back, to get back into the thesis. It might be wise to kind of save your ‘state of brain’ so that once you get back you can get back into it again fast. It might also be nice to write down some questions that you struggle with at the time right before going on vacation.*
- Last questions
    - Who can I ask for help?
    s*ome teachers on the 9th floor (e.g. S. Gehly, E. Mooij, J. Heiligers), phd students*
- Course evaluations Q3
- Other stuff we talked about
    - Notes to self
    *Kevin advices to keep a document with some notes to self on what things I learned or what decisions I made. I can maybe also find stuff about this in the A-Z book or the research methodologies?? It’s good to sometimes zoom out a bit to the helicopter view to see if I’m on the right path.
    I think for reading papers it’s good to make notes on what things I found in the paper so that I can look back at those notes and quickly find where I have read what information, instead of vaguely remembering that I have read something somewhere but not being able to find it. Also Reference Management ****Software (RMS) with tags can help with this*
    - Logbook
    *Kevin wants me to keep a high level log book. Basically a document in where I keep the hours per day, not even necessarily what activities I did, although that is also possible.*
    - Literature study purpose
    *The literature study should be like a handbook for the rest of your research.*

### Meeting 7: Week 6 (24-06)

### Agenda

- AI image
- Planning
- Findings so far
    - Learned more about CNN, RNNs, Regularization
        - *send paper to Kevin*
    - Research meth. and PhD A-Z book
        - Interact with literature and read deeply and broadly
    - Reading deeply
        - Papers by Raissi et al. & Cuomo et al.
        - Keeping the overview
- Planning next weeks
    - Focus on literature and less on skills for now
    *yes, agreed, this is necessary to finish in time with literature*
    - Start writing in a week
- Vacation → spontaneously
*That’s okay, just tell as early as possible*
- Next meeting - 4 july?
*Is planned*
- Other stuff we talked about
    - Vallado author 
    *has a book named FoA as well, this might be nice for Gravity assist maneuvers literature, maybe Wakker is not the best resource for this*
    - Add V&V to literature study
    *How will you trust the numbers that come out of your simulations?  Compare to other studies/papers? not entirely sure how to V&V my results atm… maybe look at how Thomas did this, I think Tommy Kranen in his literature study (Kevin, during the meeting, showed his literature study on his laptop, but did not share that file with me so I don’t have that) also incorporated a part on V&V*
    - **NARROW DOWN**
        - **I should have already been a bit further than I currently am**
        - **During the literature study you have to ask yourself continuously whether things you are reading help you with reaching your goal / solving the research question!
        ⇒ Once you have a basis, which I do have (on PINNs at least by now) this should be easier though, so do narrow down now**
        - **Don’t just read stuff, critically evaluate whether it is useful**
        - **You simply can’t pursue everything, evaluate as quickly as possible whether something directly will help you reaching your goal. Maybe start the other way around → Ask yourself: what do I need now to reach my goal**

### Meeting 8: Week 7 (03-07)

### Agenda

- AI image
- Literature search OCP & PINNs
- Indirect vs direct methods
- [Kolmogorov-Arnold Networks (KANs)](https://www.notion.so/Kolmogorov-Arnold-Networks-KANs-60954233e39c49b792f9e98e403f4d0c?pvs=21)
    - Adv/Disadv. → maybe do it as a potential extra
    *⇒ completely agree with this*
    - 0.5 day of extra research?
        - Better understanding of (dis)advantages vs normal FFNNs
        - Better understanding of how to implement it in Python?
        *⇒ Agree with spending 0,5 to 1 day on this. Do make sure you have clear for yourself what you want to get out of this when spending this 0,5 to 1 day on this. But what is stated above this sentence basically covers that.*
- Thomas’ [recommendations](https://www.notion.so/Physics-Informed-Neural-Networks-for-Designing-Spacecraft-Trajectories-by-Thomas-Goldman-0a9a0ee1280c40c886a325e32986d6cc?pvs=21)
    - Having a more tailored network architectures
    *⇒ You have to do this any way, when applying a PINN to a gravity assist, so this is not a ‘nice to have’, but a necessity.*
    - Adaptive loss weights scheme (to more efficiently navigate the loss space)
    *⇒ Look out that this is not so much work that it could be a thesis on it’s own*
    - Adaptive sampling of collocation points
    *⇒ Look out that this is not so much work that it could be a thesis on it’s own*
- More overarching need to narrowly constrain the work
*⇒ With our current plan of having as a basis: implementing a PINN in the same way Thomas did it to a gravity assist scenario + having a few ‘nice to haves’ like implementing KANs,  without to much extra’s, we feel like the scope is manageable*
- Planning literature study and vacation
*⇒ Ending the literature study on August 9 is okay. That would be approximately equal to a 9 week literature study*
- Have a look at other literature reports?
    - *Things to include in gravity assist literature report*
        - *What coordinate systems are there*
        - …

### Meeting 9: Week 11 (29-07)

### Agenda

- AI image
*⇒ Forgot it :(*
- Kolmogorov Arnold Networks (KANs)
    - [(dis)advantages](https://www.notion.so/Kolmogorov-Arnold-Networks-KANs-60954233e39c49b792f9e98e403f4d0c?pvs=21) w.r.t. normal FFNNs
    - Show KAN paper section 3.4 results (KAN + PINNs)
    - KINNs
    - Programming
    *⇒ showed my findings, we agreed that implementing this could take a whole thesis… but maybe, if I have time, some results can be generated.*
- Gravity assist problem to solve
    - 1st idea → doesn’t make much sense → within SOI 2 body problem
    *⇒ Could do this as a first (easier to implement) step after which it can be extended to*
    - 2nd idea → [see drawing] Determine if states are heliocentric or w.r.t. GA body before calculating loss
    - 3rd idea → divide in 2 phases,1 before GA, calculate GA phase analytically, one phase after GA, and have a loop (so first calculate phase 1, then optimize phase 2 with initial conditions derived from that phase 1)
    - 4th idea → proximity measure + range of setting initial conditions for 2nd phase
- Discuss exact deliverables of Research Proposal Review [https://brightspace.tudelft.nl/d2l/le/content/623070/Home](https://brightspace.tudelft.nl/d2l/le/content/623070/Home)
- Literature report contents, discuss whether it’s ‘complete’ → send to Kevin afterwards for checking?
- General schedule of Thesis (image I couldn’t find last meeting)
- Personal schedule (excel sheet), discuss plan for next 12 days (or 14 incl. next week’s weekend)
- Artemis presentation at MSc kick off day
*⇒ mail education AE*

### Meeting 10: Week 12 (07-08)

### Agenda

- AI image
- Pyramid Thinking structure
*⇒ Had a nice discussion on this and showed the difference of the contents before and after applying the structured thinking approach*
- V&V detail
*⇒ It’s indeed okay to not have that much details already, but do come up with idea’s. Look what ways of Verifying and Validating are out there.*
- Tight on time!
*⇒ It’s okay to skip e.g. ‘Novel Ways of Incorporating Physical Laws and Boundary Conditions ‘. Just make sure you do include writing why the section is useful (why the idea of looking for those Novelties is useful)*
- MSc kick off presentation [Artemis related]
*⇒ Mail Bart root for presentation
⇒ Email Kevin about participating at the BBQ*

### Meeting 11: Week 17 (12-09)

- Literature review
    - Feedback
    ⇒ *Pretty elaborate feedback report is send by Kevin via email, have a look at this for next meeting and list questions that you have + things that you want to discuss with Kevin*
    - 9.5 week literature study + vacation → early midterm…
    *⇒ Yes, literature study took a bit longer, let’s not reschedule midterm, but **let’s park Research activity 3 for later and start with “Research activity 4: Solving subproblem 1” right after finishing Research activity 2**, since Research activity 3 can be seen as ‘nice to haves’ as well*
    - Only 5 out of 7 allowable weeks of vacation planned for now (3 wks in summer, 2 weeks Christmas)
    *⇒ In principle I should not have to ‘sacrifice’ 2 weeks of vacation, but since the literature study took a bit longer, some extra time might come in handy. Let’s just see how things progress. Also I tried to explain that I have now planned 5 weeks of vacation in total (3 in summer, 2 during Christmas) and **I should still send Kevin the exact vacation days since Kevin only has registered 8-21 July now***
    - Back to 80 % FTE
    *⇒ just mentioned this*
    - Structured Thinking video Reference okay?
    **⇒ *Remove reference to Barbara Minto***
    - Forgot literature on learning rate schedules and adaptive weight scheduling
    *⇒ don’t take too much time for this, There is no time for redoing (part of) the literature study, have a quick look at some research and decide to implement new stuff only if it’s easy and has a high chance of it being worth it. Furthermore if you notice stuff that is harder to implement and not sure if it’s worth it, you can always mention that and potentially make it a recommendation at the end of the research*
- Mid-Term deliverable and mid-term planning (bit early since lit study took longer)
*⇒ **Mid term date stays the same, deliverable will be a presentation with results, doesn’t have to be pretty, important is to communicate results clearly to be able to assess as well as possible what the current state is and how to proceed,** you can start writing the report before this time, but concept report doesn’t have to be a deliverable.*
- Change green light meeting to 14 march?
*⇒ Check if this is actually possible, Kevin honestly doesn’t mind, but the kick off form has already been signed so check what the rules are and if this is actually possible.*
- Progress Research phase 1
    - Software exploration
        - TensorFlow + Keras, DeepXDE → customizability important
        *⇒ Fine, choose something you are comfortable with.*
    - Basic PCNN model
        - Building it peace by peace (+ simple circular orbit)t
        *⇒ Agree with this approach, also having this toy example might be useful for verification of  every single extension, so see if you could report stuff on that (is not necessary though, Thomas also verified the workings of his model with e.g. a Pendulum, of which nothing was reported)*
        - Training times: custom training loop vs using built in (optimized) functions
        *⇒ don’t recall what was discussed here…*
        - Show results
        *⇒ Happy to see first results have been obtained. Furthermore: **would be nice if you make a Github repository so I can scroll through it a bit***
        
        *In general, Kevin says progression is as expected and I am well on my way, keep in mind we’ll park research activity 3 for now though.*
        

### Meeting 12: Week 19 (25-09)

- AI image
- Literature review items to discuss
*⇒ discussed 2 minor items, the one about the previous research being done → should probably rename the section name*
- Results
***⇒ update plan (not planning) for next meeting***
    - Show loss evolutions
    *⇒ maybe the objective is 10^-16 because thrust is always maximum and then you subtract some `u_value`with a similar `u_value`which then equals ~zero, divide this by `Isp*g0` and you have ~0/`Isp*g0`  which might give you the 10^-16, or zero?*
    - Perturbed time grid problems
    *⇒ Explained the gradient hypothesis and that I’m going to try to see if that causes the problems (grads become larger when $\Delta t$’s become larger), no further input from Kevin’s side.*
- Verification of Basic PCNN (bundled vs parallel networks)
*⇒ Reproducing the box plot in the appendix might be necessary in order for verification to be convincing enough.*
- Training times → GPU training and Delft Blue
*⇒ Not surprised by it, happens often, agree you should look into this!*
- Other feedback that I was missing: Doubts about accuracy of the method and feedback about subproblems to solve?
***⇒ postponed to next meeting***
- GitHub Repo, green light rescheduling, mid-term arrangements → will go after this for next meeting
*⇒ Mentioned this very briefly*
- Q’s
***⇒ postponed to next meeting***
- Optional: Custom training vs built-in functions TensorFlow? What is happening? (API, asking other people, …)
*⇒ Showed Kevin the problem, he doesn’t have an immediate answer to it (which was expected), could be a red flag though that I haven’t seen anybody use it on the internet, everybody uses a custom training loop (so just a ‘for loop’).
  → For now: I will first look into GPU training and using Delft Blue, maybe this increases training times already significantly enough to not have to solve this problem.*

### Meeting 13: Week 21 (09-10)

- AI image
- Plan update
- Points of last meeting that we didn’t finish (see **Blue**)
- GPU training laptop
*⇒ Discussed the fact that my laptop doesn’t have a CUDA compatible GPU*
- DelftBlue
*⇒ Told Kevin that I am now able to run my code on DelftBlue*
- Mid-term arrangements (discuss what is expected)
*⇒ maybe try to see if extra people can be invited (koen?), book meeting room 6 (does it have a screen?), take 1.5 hrs*
- Results
*⇒ Showed the results of the working basic PCNN model for the first time*
- Q’s

### Meeting 14: Week 22 (18-10)

- AI image
- Plan overview
*⇒ At midterm I will show the complete verification of the Basic PCNN model and the first results of solving subproblem 1, V&V of subproblem 1 will only be done after midterm.*
- V&V results
    - Still implement the metrics + mass comparison
    
    *⇒ showed the results
    ⇒ had a discussion about whether there could be other ways to verify the Basic_PCNN model. Finding other research than Goldman’s to verify stuff with would be even more convincing, because verifying with more things is always better.*
    
- GitHub Repository
    - Rewrote my TensorFlow code to DeepXDE (while still using TensorFlow as the backend)
    *⇒ discussed how much faster DeepXDE is (1 PCNN with 56000 epochs can be trained in a bit less than 1 minute)*
    
    Also discussed that I’ve rewritten parts of the DeepXDE library, so in order to run my code you should install my version of DeepXDE. Therefore, **make a user manual/guide/instructions, first of all for Kevin to be able to also run the code and play around with it, but also potentially for other people and future graduate students that want to be able to continue the work.**
    
- Minutes of meeting
⇒ Yes, it would be nice if Kevin has access to some minutes of the meeting so that he can also remember what has been discussed **so make a document on GitHub for this (check if my repo is private though)**
- Other feedback that I was missing: Doubts about accuracy of the method and feedback about subproblems to solve?
*⇒ This is indeed an important thing to consider. Interesting things to try is also to extend the number of epochs to see if the accuracy can easily be increased by just having more epochs. Furthermore if accuracy is limiting maybe the problem could be broken down into multiple optimization problems, but it would be very interesting if the SOI entry point is an optimizable parameter. Imo it would make the whole thesis results a lot less interesting if it wasn’t the case so try to find a way of including that*
- Q’s

### Midterm: Week 24 (31-10)

Notes I taken after the meeting:

- Use beta value to optimize gravity assist? Kevin and Onur sounded positive about this. Beta value should be equal to zero, have a look at this beta value and figure out how exactly to use it
- Try gigantic NNs and train them for a very long time to see what happens, on Delft Blue
- Try to constrain the mass so that's it's not allowed to increase
- Thijs said using ReLU could be really worth it, it’s one of the most used activation functions nowadays
- Thijs said using different activation functions per layer could help
- Thijs said using adaptive weight sampling could be interesting

- Some validation/ comparison to a real life scenario is missing, you need this to have a feeling on how accurate your solutions are
- Onur was expecting some coast phases in the optimal solution, whereas in my solutions that doesn't happen, try to find optimal solutions with another optimization method to see if that's indeed what we should expect in an optimal solution. Not sure if this needs to be the case though...
- Lambert arc's, ...
- Koen said less spread when using PFNN's make sense, because it's less flexible than FNN’s without separate NNs per output
- 0.00001 AU instead of 0.0640 could be due to the fact that my TOF is way less in subproblem 1
- Look at whether using the lowest values of the metrics can be done, since there might be trade offs in there, the lowest values don't always happen at the same epoch!

### Meeting 15: Week 26 (15-11)

- AI image
- Plan overview
- Recap mid-term
    - Change previous GA Scenario (BepiColombo) to something else, for now EVEMJ
    *⇒  Kevin agrees that indeed it might be more relevant to compare the PCNN results to other preliminary trajectory optimization methods. It might be interesting to see how the fuel used compares to some real life mission though, but for verification purposes real life data is not that suitable, because of the assumptions (2D restricted Two-body problem) made by my model. I’ve also found a [paper](https://arc.aiaa.org/doi/10.2514/1.13095) where an EVEMJ case is optimized and two-body motion between GAs is assumed . Furthermore the [Hodographic shaping method in combination with a Simple Genetic Algorithm (SGA) algorithm as the optimizer](https://docs.tudat.space/en/latest/_src_getting_started/_src_examples/tudatpy-examples/mission_design/hodographic_shaping_mga_optimization.html) can be used for comparison/validation.*
    - Using ReLU (for $\theta$, and $r$) in combination with PFNN 
    *⇒ showed that using this actually decreased performance and **TODO: I am still going to try to understand why***
    - Using different activation functions per layer
    *⇒ showed that using this did not necessarily increase performance, similar results for when using* `"relu", "relu", "relu", "sin", "sin", "sin"` with FNN
    - Use tanh activation function (each layer)
    *⇒ Same results obtained. Not more accurate. tanh is similar to sin so this was expected*
    - Non increasing mass constraint
    *⇒ Same results obtained. Not more accurate. PCNN is able to recognize mass should not increase, so this is good*
    - Try without mass as NN output
    *⇒ works, it has **a bit faster runtime**, is **not more accurate** and **requires more runs per 1 successful run.** For now it’s therefore decided to leave the mass output in there since no more accurate solutions are obtained and trading more amounts of runs needed for a bit more efficiency (cpu time) is considered not worth it.*
    - Longer training time
    *⇒ could increase accuracy a tiny bit. Tried 141000 instead of the regular 51000 iterations and accuracy was a tiny bit better. Also tried it for 1 000 000 iterations, because why not, then chances of an exploding gradient become larger and this became the limiting factor here. **TODO: I could still try to do this in combination with clipping the gradient values though to prevent them from exploding.***
- Code manual GitHub repo
*⇒ Showed this*
- Results subproblem 1
*⇒ Very briefly showed this*
- $\beta$ value to optimize
*⇒ This value describes the orientation of the GA hyperbola. If it aligns with the velocity vector of the GA planet, the change in energy is maximized. So with this we have a way of telling the PCNN how to optimize the energy change. **TODO: have a look at how to use this**.
⇒ Furthermore this can also be used to verify the hyperbola part of the trajectory. You have a relation between the energy change and the orientation of the hyperbola, you can check if this relation is satisfied by the PCNN.*
    
    $\Delta \mathscr{E}=\frac{2 V_t^{\prime} V_{\infty_t} \cos \beta}{\sqrt{1+B^2 V_{\infty_t}^4 / \mu^2}} \quad ; \quad \Delta\left(\frac{1}{a}\right)=-\frac{4}{\mu_S} \frac{V_t^{\prime} V_{\infty_t} \cos \beta}{\sqrt{1+B^2 V_{\infty_t}^4 / \mu^2}}$
    
- Additional midterm feedback Kevin
*⇒ Something I could improve on is the way I communicate/present my results. I could have more structured arguments for this (and sometimes better visuals on the slides). Sometimes I presented something and I did not explain well enough how I came to that result or conclusion. Then the listener kind of has to just believe what I’m saying, whereas more structured arguments can help communicating this is a more convincing way. Coming up with these ‘structured arguments’ can also help you to not jump to conclusions too early.*
- Plan next week
*⇒ continue with subproblem 2, don’t spend too much time on V&V of subproblem 1 since I’m currently not sure what results of subproblem 1 I will include in the thesis report. Why have we done subproblem 1 at all then? To individually check the effect of all of the above things I tried on the performance and/or accuracy of the basic PCNN. Those were all pretty straight forward things to try that did not take too much time, but nothing seemed to improve performance really significantly… If this were the case this could be incorporated in solving subproblem 2. In order to check the effect it was better to try it on the basic PCNN model first, before extending and modifying the basic PCNN model and then trying it.*

### Meeting 16: Week 29 (05-12)

- AI image
- Plan overview
- *⇒ Bernd Dachwald has done some research on GAs and NNs in the 80s/90s, might be fun to look at ← **TODO***
- SP2: Technique 1
    - Constraint layer
    - 8th output

*⇒ see if changing the magnitudes of r_ga can help for convergence*

- SP2: Technique 2
    - 2 extra outputs $\delta$ and $\beta$
    - Constraining $r$, $\theta$, $v_r$, $v_{\theta}$ right before and after GA

*⇒ Could be interesting!*

- EOMs check
*⇒ You could verify this with Wakker chapter 4 and/or Tudat. When doing it with tudat you could e.g. numerically integrate with your EOMs and compare it, that could then serve as the verification of purely the EOMs*
- $\dot{r}=v_r \\\dot{\theta}=\frac{v_\theta}{r} \\ \dot{v}_r=\frac{v_\theta^2}{r}-\frac{\mu}{r^2}+a_r^{\text {perturb }}\\ \dot{v}_\theta=-\frac{v_r v_\theta}{r}+a_\theta^{\text {perturb }}$
- $\begin{aligned}& \mathbf{a}_{\text {perturb }}=-\frac{G M_{\text {Mars }}}{\left\|\mathbf{r}_{\mathrm{sc}}-\mathbf{r}_{\text {Mars }}\right\|^3}\left(\mathbf{r}_{\mathrm{sc}}-\mathbf{r}_{\mathrm{Mars}}\right)\\ & a_r^{\text {perturb }}=\frac{\mathbf{a}_{\text {perturb }} \cdot \hat{\mathbf{r}}}{\|\hat{\mathbf{r}}\|} \\ & a_\theta^{\text {perturb }}=\frac{\mathbf{a}_{\text {perturb }} \cdot \hat{\theta}}{\|\hat{\theta}\|}\end{aligned}$
- General discussions
*⇒ Keep Kevin in the loop. Keep making sure all the work you do keeps being verified sufficiently to make sure the results can be trusted with high confidence.*

### Meeting 17: Week 30 (13-12)

- AI image
- Manual basic PCNN
*⇒ Kevin hasn’t tried to run it yet, but he did have a look at it*
- SP2: Technique 1 (8th output and adjusted constraint layer)
    - Too many times I get nan values
        - Tried: grad clipping, underflow, overflow → now checking every value to exactly pin point where it happens → could help understand what is happening…
        *⇒ Maybe start asking people around about this now*
    - For this, maybe start with no extra point in time domain and only after convergence has already happened a bit, add time points to prevent the early nan values
    *⇒ What could work for this is to make a custom for loop that trains the model for a while, then stops and then continues training it with a newly sampled time domain. **← TODO***
- SP2: Technique 2
    - 2 ways of doing it:
        - 2 extra **outputs** $\delta$ and $\beta$
        - 2 extra **trainable parameters** $\delta$ and $\beta$
        - Adding a dynamical loss term OR constraining $r$, $\theta$, $v_r$, $v_{\theta}$ right before and after GA

*⇒ I was implementing this technique as we speak so we did not discuss it too elaborately*

### Meeting 18: Week 34 (09-01)

- AI image
- Planning
- Technique 1:
⇒ *nan* values solution → transfer learning
- Technique 2:
    - Extension of outputs
    - Trainable parameters $\delta_{ga}, r_p, v_{sc}^-$ → constrain initial state leg 2
    - Added GA loss term
    - Nondimensionalization leg 2 → same time domain input so scale time with `(TOF_MC_days/TOF_EM_days)`
    - Lot’s of debugging and playing around with loss weights and different initial values for $\delta_{ga}, r_p$
    - Results: delta→0
    *other comments
    ⇒ Would shaking delta_ga be an idea to make sure it doesn’t stay at the value it converges to after only a few iterations already.
    ⇒ MBH monotonic basin hopping as an alternative to LR scheduling ← **TODO**: look at this briefly
    ⇒ sundman transform for nondimensionalization of parameters ← **TODO**: look at this briefly
    ⇒ seaborn plots for determining causalities
    ⇒ simulating GA part as R2BP as well, so basically having 3 legs ← **TODO**: Why haven’t I done this?*
- Thrust range question
*⇒ allowing 360 degrees of  thrust direction around the S/C is fine*
- Planning
- ***TODO**: write email to thesis duration committee and read whole Brightspace space about thesis*

### Meeting 19: Week 35 (17-01)

- AI image
- Planning
- T2 results
*⇒ Try training it for an even longer time than I already tried ← **TODO***
- T3 results
*⇒ Check whether `r_p` is w.r.t surface of com of Venus ← **TODO***
- Started writing
- From $\Delta V$ to fuel used:
*⇒ this is not the way to check fuel, you have to integrate the thrust profile of course. ← **TODO***
    - $\Delta v=I_{s p} \cdot g_0 \cdot \ln \left(\frac{m_0}{m_f}\right)$
    - $m_{\text {fuel }}=m_0-m_f$
    - $m_f=m_0 \cdot e^{-\frac{\Delta v}{I_{\text {sp }} g_0}}$
    - $m_{\text {fuel }}=m_0 \cdot\left(1-e^{-\frac{\Delta v}{I_{s p} \cdot 9_0}}\right)$
- Planning
⇒ **TODO**: Have a quick chat with academic counsellors to see what possibilities to extension there are
⇒ **TODO**: Send thesis duration committee an email about the uploading of the thesis report being at the end of week 47

### Meeting 20: Week 37 (28-01)

- AI image
- Plan
**TODO** → send kevin screenshot of (nominal) plan incl. dates of milestones
- `r_p` constraint (w.r.t. surface of Venus)
    - Still to do: analysis on scaling and loss weight (grid searches?)
- Integrating HS thrust profile + HS vs PCNN comparison
*⇒ would we have expected this? ← **TODO** look at whether this was expected or not, I might have done that already, but in that case I forgot.
Furthermore, always include this step - checking whether the results was to be expected or not - in your work. Never just blindly assume the numbers coming out of your simulation are correct. I realize this, but it doesn’t hurt to note it down again.*
- RAR
    - No lower loss terms or metrics, maybe more efficiency (faster convergence) though
    *⇒ Still do a proper simulation so that you can report on it and draw a conclusion (and have proper evidence supporting that conclusion)*
- SP3 efforts
    - t_ga to fuel usage / delta V
    - Varying t_ga → Hoping for quick decrease in dynamics loss terms
    - Question: Other ways of evaluating fuel usage/ Delta V for specific initial/final state and TOF → solving Lambert problem?
    *⇒ not sure if there is such a thing. Lambert arc / simpson flanagan could work?? but another problem might be to connect the positions of the arcs and that’s a problem which you might now want to include in the scope of this thesis*
- Contacting ‘experts’
    - Show questions to discuss
- More writing!
- Thesis extension possibilities
⇒ Chat with academic counsellors
⇒ Week 47 is indeed uploading into repository date confirmation

### Meeting 21: Week 38 (06-02)

- ~~AI image~~
- Plan
- HS thrust profile: $\Delta V$ with ideal rocket equation vs integrating thrust profile: 435 kg vs 415kg **(~5%)** or 43.5kg vs 41.5kg **(also  ~5%)**
*⇒ would we have expected this? Yes because now mass changes throughout trajectory*
- Residual based adaptive sampling of training points
    - RAD best performance
    - Same seed, M=1000,200,100,50,10, with and without resampling comparison
    - Setting up big simulation
- Spend time on:
    - J.B. Stiasny (conversation + document)
    - DelftBlue
    - Writing
    - Variable t_ga (sp3)
- Mamba (built on top of conda)

### Meeting 22: Week 39 (14-02)

- AI image
- Writing
    - Question about conciseness of the paper
    *⇒ A good technique would be to keep it as short as possible, then add things if you still have space left. The other way around; cutting things, is probably harder to do.*
    - Feedback already?
    *⇒ short feedback cycle is okay, make sure to ask precisely what type of feedback you want.
    ⇒ finding other people to read your text is good too, maybe let every person only read a few pages*
- J.B. Stiasny (conversation + document)
*⇒ look at outputs before transformation*
- Debugging
*⇒ Told Kevin I spent like 1.5 days doing this.*
- Alternative constraint functions
*⇒ showed results of other constraint functions*
- plan
*⇒ Discussed the plan. **TODO**: ask academic counsellors for advice on how much extension to ask for*
- Next up:
    - Get DefltBlue working again
    - Mostly writing - V&V

### Questions for Supervisor

- **The paper needs to be concise, but should you explain why you made certain choices (like why polar coordinates and not cartesian coordinates, maybe very briefly?**
*⇒ A good technique would be to keep it as short as possible, then add things if you still have space left. The other way around; cutting things, is probably harder to do. Sometimes explaining a choice can be desired, e.g. if it’s very important to the research.*
- Realistic ranges for thrust directions? 0-360 for discovering all possible trajectories? If you have 4 thrusters in prograde, retrograde and radial direction you could achieve this right?
*⇒ allowing 360 degrees of  thrust direction around the S/C is fine*
- Do you have any other literature on anything you still want to recommend?
*⇒ Vallado - FoA book*
- There’s not much argumentation in Thomas’ paper for the choices he made, is that included in his literature study? Might be nice to share this then with me..?
Indirect method vs direct methods
Direct method is easier to implement
Indirect method probably requires some extra studying. Came across a lot of terms that I don’t understand yet: (Extreme) Theory of Functional Connections ( (X)-TFC)), Pontryagin minimum principle
*⇒ I suspect that at least part of the consideration could have been a practical one: the direct method is easier to implement and one can, within the confines of a thesis project, focus on other aspects of the research which may be more important or key to the proposed work. It would be my moderately strong advice to focus on the direct method. Digging into the indirect method is, as you also state, much more mathematical and takes significantly more effort.*
- Just to check: I am planning on including a section in literature report on direct vs indirect methods (to explain why the direct method is chosen) and KANs (because they might be used)
*⇒ Yes, those are good reasons*
- Do you maybe have the 5th edition pdf of FoA and Applications of Vallado?
*⇒ No, only fourth edition*
- Idea to do my V&V
    - Verification (have we calculated it right) → Numerical integration of the initial state + control profile to see whether numerical values match?
    - Validation (have we calculated the right thing/ Is the intended purpose fulfilled?) → Calculate trajectories with other traditional optimization methods to see whether my optimal values are competing or better
    
    *⇒ Think I have discussed this in a meeting, took no notes on the answers from Kevin’s side then I guess…*
    
- Mission to mimic (New Horizons, Psyche, Voyager?) Mission maybe should really not be possible without GA to force PINN to use the GA, e.g. Psyche might be possible without GA and extra fuel?
*⇒ check whether data is available for the missions, then you could also use it for validation, this could be another reason to choose a certain mission*
- Does the literature study need to be updated? when? It won’t change the rest of my thesis ofc, and I remember you once telling me we’re not revisiting it…
*⇒ Do not revise immediately, is not best use of time, literature study will be added to the thesis report at the end, so it should be cohesive with that though, e.g. if a major change has occurred (change in research question), it may need to be adjusted. Literature study is a snapshot at a certain moment in time of the research, and it is normal that things do not go exactly as planned in the literature report*
- Tips for when stuck with very specific questions?
*Discussed various tips for when stuck (asking around, find people that work on somewhat similar problems, rubber duck effect, remember having this is absolutely normal)*
- Would it be an idea to not limit the maximum allowable thrust
⇒ Well if I have it at 1.0 N, it still becomes like 0.1 or 0.05 on Average, at least in my T3 technique so I guess having it at 100N or unlimited works the same..? worth a shot though

### Questions for Thesis boost day

- Where to put references? Once at end of sentence? What if you have a whole bunch of facts stemming from 1 ref, then multiple refs of it?
    - Should you reference sections if referencing a whole textbook?
- How often do you spell out the whole word before using acronyms (like once in every chapter and then after that no more? Or once in a whole report?)

### Questions for Thomas

- Have you once initialized the weights and biases not randomly? I could use that for verification
*⇒ He uses a seed for this*
- Why is the perturbed sampling needed?
*⇒ Ik heb inderdaad weinig redenen voor de exacte manier waarop ik de time batches sample. Die manier heb ik gelezen in dat artikel waar ik aan refereer en dat heb ik toen gewoon overgenomen. Mijn logica was dat als de thrust output een beetje een grillerige vorm zou hebben, dat de random tijd samples voor verschillende training epochs ervoor zouden zorgen dat alle momenten in de tijd "behandeld" worden voor die integraal, dus ook lokale detail. Maar ik heb verder geen structureel onderzoek gedaan naar het effect van de time sampling. Ik was ook al inderdaad tot de conclusie gekomen dat de hele methode niet zo gevoelig is voor de sampling methode, vandaar dat ik daar niet verder op gefocussed heb. Verder heb je inderdaad gelijk en is het natuurlijk in het algemeen ook handig om overfitten te voorkomen. Maar geen specifieke reden dus.*
- Why is mass as an output of the NN needed, if you already have the thrust magnitude as an output. It should be possible to omit this and calculate the mass per *t* based on the predicted applied thrust.
*⇒ you could try this, what happens if mass becomes negative though..? Also see email for more elaborate answer.*