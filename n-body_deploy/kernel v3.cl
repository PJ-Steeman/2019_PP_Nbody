#define DELTA_TIME 1.f
#define GRAV_CONSTANT 1
#define MASS_OF_SUN 2
#define DISTANCE_TO_NEAREST_STAR 50

//Atomic Add (atomische operaties worden gebruikt om race conditions tegen te gaan)
//Ze worden uitgevoerd zonder tussenkomst van een andere thread
inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal,newVal.intVal) != prevVal.intVal);
}
//union gebruikt in atomische operaties (sizeof(float3_) = sizeof(float3))
typedef union
{
    float3 vec;
    float arr[3];
} float3_;
//de snelheids berekening via AtomicAdd
__kernel void snelheid(__global float3 *gpu_pos, __global float3_ *gpu_speed)
{
        const int i = get_global_id(0);
        const int j = get_global_id(1);
        const int mass_grav = GRAV_CONSTANT * MASS_OF_SUN * MASS_OF_SUN;

        if (i == j)
        {
            return;
        }

        float3 pos_a = gpu_pos[i];
        float3 pos_b = gpu_pos[j];

        float dist_x = (pos_a.s0 - pos_b.s0) * DISTANCE_TO_NEAREST_STAR;
        float dist_y = (pos_a.s1 - pos_b.s1) * DISTANCE_TO_NEAREST_STAR;
        float dist_z = (pos_a.s2 - pos_b.s2) * DISTANCE_TO_NEAREST_STAR;


        float distance = sqrt(
                dist_x * dist_x +
                dist_y * dist_y +
                dist_z * dist_z);

        float force_x = -mass_grav * dist_x / (distance * distance * distance);
        float force_y = -mass_grav * dist_y / (distance * distance * distance);
        float force_z = -mass_grav * dist_z / (distance * distance * distance);

        float acc_x = force_x / MASS_OF_SUN;
        float acc_y = force_y / MASS_OF_SUN;
        float acc_z = force_z / MASS_OF_SUN;

        AtomicAdd(&gpu_speed[i].arr[0], acc_x * DELTA_TIME);
        AtomicAdd(&gpu_speed[i].arr[1], acc_y * DELTA_TIME);
        AtomicAdd(&gpu_speed[i].arr[2], acc_z * DELTA_TIME);
}
