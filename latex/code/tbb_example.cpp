 return parallel_reduce(
        // array von floats, mit Blöcken der Größe grainsize
        blocked_range<float*>( array, array+n, grainsize),
        // Intialer Wert für Vereinigung
        0.f,
        //Transformation
        [](const blocked_range<float*>& r, float init)->float {
            // Iteriert über jedes Element des Blocks r
            for( float* a=r.begin(); a!=r.end(); ++a )
            // und addiert das transformierte Element zur Gesamtsumme
                init += do_something(*a);
            return init;
        },
        // Vereinigung
        []( float x, float y )->float {
            return x+y;
        }
    );