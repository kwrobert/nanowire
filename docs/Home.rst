Introduction
------------

Hello there! If you are reading this, you're probably trying to figure out how
to use my code. Hopefully, this documentation will be helpful, but it is still
a work in progress as is the actual code.

I'll be continually adding features to the development branch. I'll only merge
those changes into the master branch when things are stable. However, I can't
guarantee there won't be API changes, changes in function names and signatures,
changes in data organization, etc.

Overview
--------

This code uses a C library called S4 developed at Stanford
(http://web.stanford.edu/group/fan/S4/index.html). This library implements the
Rigorous Coupled Wave Analysis algorithm for optical modeling of 3D devices
(although I think it's capable of 1 and 2D simulations as well). As of the
writing of this document, this library a wrapper around S4 to automate
constructing geometries, running parameter sweeps, etc. In the future, I hope
to abstract away the backend algorithms that actually solve Maxwell's equations
behind a unified interface. This way, the user can configure at runtime whether
they want to use an RCWA, finitite difference, or transfer matrix solver (which
each come with their own limitations and advantages).

This module is completely controlled by an input text file in valid `YAML
<http://yaml.org/>`_ format. If you've never heard of it, YAML is a data
serialization format like JSON or XML with the unique advantage of being very
human-readable. This file is used to configure the runtime behavior of all
scripts and define the parameters of the physical system you are analyzing. If
this file is malformed or not present, basically nothing runs. 

Some of you out there might be upset because you are unable to define your
system programatically. Your objections are heard, but this approach allows all
input files to be version controlled, hashed, and otherwise compared for
equivalence and duplication. Although slightly less flexible than a
programmatic interface, it facilitates and ensures reproducibility. If I get
enough complaints and some constructive criticisms/suggestions, I might add a
programmatic interface in the future. 

Questions/Concerns?
-------------------

Email me! If you are actually going to use this software to do science, that's
great! I'm a huge believer in open-source scientific software and would be
happy to fix any bugs you find, collaborate with you, or answer any questions
you may have. 
