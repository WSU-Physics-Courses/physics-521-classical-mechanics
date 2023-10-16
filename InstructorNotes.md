Instructor Notes
================

https://www.youtube.com/watch?v=yrc632oilWo&ab_channel=RealEngineering

## Perusall

I created a new course, then copied the previous documents and assignments over,
adjusting the due dates.

## CoCalc

1. Create a new course in the [WSU Courses CoCalc Project].  I don't copy old ones because
   this can create issues with participants and the shared project id.
2. Add all the students using their WSU email addresses.
3. Configure so students have 2 weeks to pay.
4. Open the shared project, add the URL to your `.cookiecutter.yaml` file, etc. and
   rebuild.
5. SSH into the shared project, copy the repos, and install:

   ```bash
   ssh smc521   # I create an alias in my ~/.ssh/config file
   pipx install mmf-setup
   mmf_setup cocalc
   mkdir .repositories
   cd .repositories
   hg clone git@gitlab.com:wsu-courses/physics-521-classical-mechanics.git
   cd physics-521-classical-mechanics
   make init
   cd ~
   ln -s .repositories/physics-521-classical-mechanics/ phys-521
   ```



[WSU Courses CoCalc Project]: <https://cocalc.com/projects/c31d20a3-b0af-4bf7-a951-aa93a64395f6/files>


