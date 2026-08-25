/**
 *
 * @file starpu/runtime_rectasks.h
 *
 * @copyright 2019-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon StarPU recursive-task support
 *
 * @version 1.4.0
 * @author Mathieu Faverge
 * @date 2024-02-18
 *
 */
#ifndef _runtime_rectasks_h_
#define _runtime_rectasks_h_

/**
 * @brief Recursive task arguments structure
 */
struct rectask_args_s {
    /**
     * @brief Context in which the subtasks are submitted. It is destroyed only
     * after a synchronization that guarantees execution of every task, so a
     * recursive task may safely retain this pointer for subtask submission.
     */
    RUNTIME_sequence_t *sequence;
    /**
     * @brief Parent task of all tasks submitted within this recursive task.
     * Used for profiling information.
     */
    struct starpu_task *parent;
    /**
     * @brief Priority of the parent task to propagate the information to the
     * subtasks if needed.
     */
    int priority;
    /**
     * @brief List of tiles used to submit the tasks. Each codelet uses a
     * different number of tiles, so the structure ends with a variable-length
     * tile array.
     */
    CHAM_tile_t *tiles[1];
};
typedef struct rectask_args_s rectask_args_t;

/**
 * @brief Allocate and initialize arguments shared by recursive codelets.
 *
 * @param[in] options
 *          Runtime options whose sequence, parent, and priority are inherited.
 *
 * @param[in] ntiles
 *          Number of entries required in the trailing tile array.
 *
 * @return Initialized recursive-task arguments.
 */
static inline rectask_args_t *
starpu_cham_rectask_args_create( const RUNTIME_option_t *options, int ntiles )
{
    rectask_args_t *rtargs;

    assert( ntiles > 0 );
    rtargs = malloc( sizeof(*rtargs) + ( ntiles - 1 ) * sizeof(rtargs->tiles[0]) );
    assert( rtargs != NULL );

    rtargs->sequence = options->sequence;
    rtargs->parent   = options->request->parent;
    rtargs->priority = options->priority;
    return rtargs;
}

/**
 * @brief Macro to specify the parent task and improve profiling information.
 */
#if defined(CHAMELEON_RECURSIVE_TASKS_PROFILE)
#define INSERT_TASK_RECTASK_PROFILE_PARAM STARPU_RECURSIVE_TASK_PARENT, (rtargs ? rtargs->parent : NULL ),
#else
#define INSERT_TASK_RECTASK_PROFILE_PARAM
#endif

/**
 * @brief Add recursive-task arguments to an rt_starpu_insert_task() call.
 */
#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
#define INSERT_TASK_RECTASK_PARAMS(__name__)                              \
    STARPU_RECURSIVE_TASK_FUNC, (is_rectask ? starpu_cham_is_recursive : starpu_cham_is_not_recursive ), \
    STARPU_RECURSIVE_TASK_GEN_DAG_FUNC,     cl_##__name__##_rectask_func, \
    STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, rtargs,                       \
    INSERT_TASK_RECTASK_PROFILE_PARAM
#else
#define INSERT_TASK_RECTASK_PARAMS(__name__)
#endif

/**
 * @brief Internal function to return true if the task will always be recursive
 */
static inline int
starpu_cham_is_recursive( __attribute__((unused)) struct starpu_task *task,
                          __attribute__((unused)) void               *args,
                          __attribute__((unused)) void              **descs )
{
    return 1;
}

/**
 * @brief Internal function to return false if the task will never be recursive
 */
static inline int
starpu_cham_is_not_recursive( __attribute__((unused)) struct starpu_task *task,
                              __attribute__((unused)) void               *args,
                              __attribute__((unused)) void              **descs )
{
    return 0;
}

/**
 * @brief Internal function to initialize request in recursive tasks
 */
static inline void
starpu_cham_rectask_initrequest( struct starpu_task *task,
                                 RUNTIME_request_t  *request )
{
    /* Register the task parent */
    request->parent = task;

    /* Propagate the rectask priority to subtasks if needed */
    request->priority = task->priority;

    /* The outer task owns the data lifetime while its partition is active. */
    request->flush = 0;

    return;
}

#if defined(CHAMELEON_USE_RECURSIVE_TASKS)
static inline void
chameleon_recursive_tile_filter( void                      *parent_interface,
                                 void                      *child_interface,
                                 struct starpu_data_filter *f,
                                 unsigned                   id,
                                 unsigned                   nchunks )
{
    starpu_cham_tile_interface_t *parent = (starpu_cham_tile_interface_t *)parent_interface;
    starpu_cham_tile_interface_t *child  = (starpu_cham_tile_interface_t *)child_interface;

    child->id         = parent->id;
    child->dev_handle = parent->dev_handle;
    child->flttype    = parent->flttype;

    (void)f;
    (void)id;
    (void)nchunks;
}
#endif

#endif /* _runtime_rectasks_h_ */
