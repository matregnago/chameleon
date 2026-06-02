/**
 *
 * @file starpu/runtime_codelet_profile.h
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon StarPU codelet profiling header
 *
 * @version 1.4.0
 * @author Cedric Augonnet
 * @author Mathieu Faverge
 * @author Cedric Castagnede
 * @author Florent Pruvost
 * @author Brieuc Nicolas
 * @author Nathalie Furmento
 * @date 2025-12-19
 *
 */
#ifndef _runtime_codelet_profile_h_
#define _runtime_codelet_profile_h_

#include <math.h>
#include <assert.h>

#define CHAMELEON_CL_CB(name, _m, _n, _k, _nflops)                                             \
    static measure_t name##_perf[STARPU_NMAXWORKERS];                                          \
    void cl_##name##_callback(__attribute__((unused)) void* arg)                               \
    {                                                                                          \
        struct starpu_task *task = starpu_task_get_current();                                  \
        /* XXX we assume square tiles here ! */                                                \
        __attribute__ ((unused)) double M = (double)(_m);                                      \
        __attribute__ ((unused)) double N = (double)(_n);                                      \
        __attribute__ ((unused)) double K = (double)(_k);                                      \
        double flops = (_nflops);                                                              \
        struct starpu_profiling_task_info *info = task->profiling_info;                        \
        if ( info == NULL ) return;                                                            \
        double duration = starpu_timing_timespec_delay_us(&info->start_time, &info->end_time); \
        double speed = flops/(1000.0*duration);                                                \
        name##_perf[info->workerid].sum       += speed;                                        \
        name##_perf[info->workerid].sum2      += speed*speed;                                  \
        name##_perf[info->workerid].n         += 1;                                            \
        name##_perf[info->workerid].duration  += duration;                                     \
        name##_perf[info->workerid].duration2 += duration*duration;                            \
    }                                                                                          \
    void profiling_display_##name##_info(void)                                                 \
    {                                                                                          \
        unsigned worker;                                                                       \
        int header = 0;                                                                        \
        for (worker = 0; worker < starpu_worker_get_count(); worker++)                         \
        {                                                                                      \
            if (name##_perf[worker].n > 0)                                                     \
            {                                                                                  \
                if ( !header ) {                                                               \
                    fprintf( stderr, "Performance for kernel " #name "\n");                    \
                    fprintf( stderr,                                                           \
                             "\tWorker  GFlop/s  delta  Nb"                                    \
                             " MeanTime(s) DeltaTime(s) TotalTime(s)\n" );                     \
                    header = 1;                                                                \
                }                                                                              \
                char workername[128];                                                          \
                starpu_worker_get_name(worker, workername, 128);                               \
                                                                                               \
                long   n         = name##_perf[worker].n;                                      \
                double sum       = name##_perf[worker].sum;                                    \
                double sum2      = name##_perf[worker].sum2;                                   \
                double duration  = name##_perf[worker].duration;                               \
                double duration2 = name##_perf[worker].duration2;                              \
                                                                                               \
                double flops_avg = sum / n;                                                    \
                double flops_sd  = sqrt((sum2 - (sum*sum)/n)/n);                               \
                double time      = duration * 1e-6;                                            \
                double time_avg  = ( duration / n ) * 1e-6;                                    \
                double time_sd   = ( sqrt((duration2 - (duration*duration)/n)/n) ) * 1e-6;     \
                                                                                               \
                fprintf( stderr, "\t%s\t%.2lf\t%.2lf\t%ld\t%e\t%e\t%e\n",                      \
                         workername, flops_avg, flops_sd, n, time_avg, time_sd, time );        \
            }                                                                                  \
        }                                                                                      \
    }

#define CHAMELEON_CL_CB_HEADER(name)                    \
    extern struct starpu_perfmodel*cl_##name##_save;    \
    void cl_##name##_callback(void*);                   \
    void profiling_display_##name##_info(void)

#endif /* _runtime_codelet_profile_h_ */
