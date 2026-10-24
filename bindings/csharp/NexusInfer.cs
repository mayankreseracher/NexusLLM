using System;
using System.Runtime.InteropServices;

namespace NexusInfer
{
    [StructLayout(LayoutKind.Sequential)]
    public struct NexusLocalMemoryStats
    {
        public ulong Tier0AllocatedBytes;
        public ulong Tier0CapacityBytes;
        public ulong Tier1HostRamBytes;
        public ulong Tier1InactiveBytes;
        public ulong BytesTransferredTier1ToTier0;
        public ulong PrefetchHits;
        public ulong PrefetchMisses;
        public double PcieBandwidthGbps;
        public double PrefixCacheHitRate;
    }

    public class NexusEngineInstance : IDisposable
    {
        private const string LibName = "nexus_engine.dll";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr nexus_create_engine(ulong tier0Bytes, ulong tier1Bytes, ulong maxKvBlocks);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void nexus_destroy_engine(IntPtr engine);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nexus_load_model_auto(IntPtr engine, [MarshalAs(UnmanagedType.LPStr)] string filepath);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nexus_prefetch_layer(IntPtr engine, [MarshalAs(UnmanagedType.LPStr)] string layerName);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nexus_submit_request(
            IntPtr engine,
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int[] promptTokens,
            ulong promptLen,
            ulong maxNewTokens
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nexus_step_continuous_batch(
            IntPtr engine,
            IntPtr outFinishedIds,
            int[] outTokens,
            ulong maxOut
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void nexus_get_memory_stats(IntPtr engine, ref NexusLocalMemoryStats outStats);

        private IntPtr _handle;

        public NexusEngineInstance(ulong tier0Mb = 512, ulong tier1Mb = 4096, ulong maxKvBlocks = 4096)
        {
            ulong t0 = tier0Mb * 1024 * 1024;
            ulong t1 = tier1Mb * 1024 * 1024;
            _handle = nexus_create_engine(t0, t1, maxKvBlocks);
            if (_handle == IntPtr.Zero)
            {
                throw new InvalidOperationException("Failed to initialize Nexus native engine.");
            }
        }

        public bool LoadModel(string modelPath)
        {
            return nexus_load_model_auto(_handle, modelPath) == 0;
        }

        public bool PrefetchLayer(string layerName)
        {
            return nexus_prefetch_layer(_handle, layerName) == 0;
        }

        public bool SubmitRequest(string requestId, int[] promptTokens, ulong maxNewTokens = 128)
        {
            return nexus_submit_request(_handle, requestId, promptTokens, (ulong)promptTokens.Length, maxNewTokens) == 0;
        }

        public int StepBatch(int[] outTokensBuffer)
        {
            return nexus_step_continuous_batch(_handle, IntPtr.Zero, outTokensBuffer, (ulong)outTokensBuffer.Length);
        }

        public NexusLocalMemoryStats GetMemoryStats()
        {
            NexusLocalMemoryStats stats = new NexusLocalMemoryStats();
            nexus_get_memory_stats(_handle, ref stats);
            return stats;
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                nexus_destroy_engine(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~NexusEngineInstance()
        {
            Dispose();
        }
    }
}
