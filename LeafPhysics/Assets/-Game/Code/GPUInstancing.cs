using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;

public class GPUInstancing : MonoBehaviour
{
    [SerializeField] private int instanceCount;
    [SerializeField] private float force;
    [SerializeField] private Vector3 gravity;
    [SerializeField][Range(0, 1)] private float upForce = 0.4f;
    [SerializeField][Range(0.1f, 1)] private float friction;
    [SerializeField] private float radius;
    [SerializeField] private float spawnHeight;
    [SerializeField] private float groundHeight;
    [SerializeField] private Vector2 positionRange;
    [SerializeField] private Vector2 scaleRange;
    [SerializeField] private Mesh mesh;
    [SerializeField] private Material material;
    [SerializeField] private Transform head;
    [SerializeField] Color[] colors;

    Color[][] instanceColors;
    private Vector3[][] velocities;
    private Matrix4x4[][] matrices;
    private VelocityUtil velocityUtil;
    private MaterialPropertyBlock mpb;

    void Start()
    {
        mpb = new MaterialPropertyBlock();
        velocityUtil = new VelocityUtil(head);

        velocities = new Vector3[instanceCount / 1023 + 1][];
        matrices = new Matrix4x4[instanceCount / 1023 + 1][];

        for (int i = 0; i < instanceCount / 1023 + 1; i++)
        {
            matrices[i] = new Matrix4x4[1023];
            velocities[i] = new Vector3[1023];
            for (int j = 0; j < 1023; j++)
            {
                var newPos = new Vector3(Random.Range(-positionRange.x, positionRange.x), spawnHeight,
                    Random.Range(-positionRange.y, positionRange.y));
                var randomRotate = new Vector3(0, Random.Range(-360, 360), 0);
                var scale = Random.Range(scaleRange.x, scaleRange.y);
                var randomScale = Vector3.one * scale;
                matrices[i][j] = Matrix4x4.TRS(newPos, Quaternion.Euler(randomRotate), randomScale);
                velocities[i][j] = Vector3.zero;
            }
        }
    }

    private void Update()
    {
        velocityUtil.Update();
        CalcuateMatricesJobs();

        foreach (Matrix4x4[] batch in matrices)
        {
            Graphics.DrawMeshInstanced(mesh, 0, material, batch, 1023, mpb, ShadowCastingMode.On);
        }
    }

    private void CalcuateMatricesJobs()
    {
        for (int i = 0; i < instanceCount / 1023 + 1; i++)
        {
            var matricesChunk = this.matrices[i];
            var velocitiesChunk = this.velocities[i];

            var matrices = new NativeArray<Matrix4x4>(matricesChunk, Allocator.TempJob);
            var velocities = new NativeArray<Vector3>(velocitiesChunk, Allocator.TempJob);

            var job = new PhysicsJob()
            {
                deltaTime = Time.deltaTime,
                force = force,
                friction = friction,
                groundHeight = groundHeight,
                headPosition = head.position,
                radius = radius,
                speed = velocityUtil.speed,
                upForce = upForce,
                baseSeed = i + 1,
                velocities = velocities,
                matrices = matrices,
                gravity = gravity
            }.Schedule(matrices.Length, 64);

            job.Complete();

            NativeArray<Matrix4x4>.Copy(matrices, this.matrices[i]);
            NativeArray<Vector3>.Copy(velocities, this.velocities[i]);

            matrices.Dispose();
            velocities.Dispose();
        }
    }

    #region PysicsJob
    [BurstCompile]
    public struct PhysicsJob : IJobParallelFor
    {
        [ReadOnly] public float force;
        [ReadOnly] public float upForce;
        [ReadOnly] public float radius;
        [ReadOnly] public float speed;
        [ReadOnly] public float deltaTime;
        [ReadOnly] public float groundHeight;
        [ReadOnly] public float friction;

        [ReadOnly] public Vector3 headPosition;
        [ReadOnly] public Vector3 gravity;

        [ReadOnly] public int baseSeed;
        [ReadOnly] public bool useSingleMatrix;

        public NativeArray<Matrix4x4> matrices;
        public NativeArray<Vector3> velocities;

        public void Execute(int index)
        {
            if (useSingleMatrix && index >= matrices.Length) return;

            var seed = baseSeed + index;
            var rnd = new Unity.Mathematics.Random((uint)seed);

            matrices[index].Decompose(out Vector3 pos, out Quaternion rot, out Vector3 scale);
            velocities[index] -= gravity * deltaTime;
            velocities[index] -= (velocities[index]) * deltaTime;
            var dist = Vector3.Distance(headPosition, pos);

            if (dist < radius)
            {
                var t = 1 - dist / radius;
                var dir = headPosition - pos;
                dir.y -= upForce;
                if (speed > 0.5f)
                {
                    velocities[index] += dir * (deltaTime * force * t * Mathf.Clamp01(speed));
                }
            }

            if (velocities[index].magnitude > 1f)
            {
                quaternion q = new quaternion(rnd.NextFloat(-1, 1), rnd.NextFloat(-1, 1), rnd.NextFloat(-1, 1), 0);
                q = quaternion.Euler(rnd.NextFloat3(-360, 360));
                rot = math.slerp(rot, q, 0.75f);
            }

            if (pos.y < groundHeight)
            {
                pos.y = groundHeight;
                velocities[index] = Vector3.MoveTowards(velocities[index], Vector3.zero, friction);
            }

            velocities[index] += deltaTime * Vector3.one;
            pos -= velocities[index] * deltaTime;

            matrices[index] = float4x4.TRS(pos, rot, scale);
        }
    }
    #endregion
}