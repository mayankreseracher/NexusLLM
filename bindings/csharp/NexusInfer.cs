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

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct NexusPlatformInfoRaw
    {
        public uint OsType;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
        public string OsName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
        public string CpuArch;
        public uint VectorIsa;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
        public string VectorIsaName;
        public uint IsUnifiedMemory;
        public uint HasTransparentHugepages;
        public uint HardwareConcurrency;
        public double TheoreticalBandwidthGbps;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string PerformanceTierDesc;
    }

    public class PlatformCapabilities
    {
        public string OsName { get; set; }
        public string CpuArch { get; set; }
        public string VectorIsaName { get; set; }
        public bool IsUnifiedMemory { get; set; }
        public bool HasTransparentHugepages { get; set; }
        public uint HardwareConcurrency { get; set; }
        public double TheoreticalBandwidthGbps { get; set; }
        public string PerformanceTierDesc { get; set; }
    }

    public class NexusEngineInstance : IDisposable
    {
        // On .NET Core / .NET 6+, "nexus_engine" dynamically resolves to
        // nexus_engine.dll on Windows, libnexus_engine.so on Linux, and libnexus_engine.dylib on macOS.
        private const string LibName = "nexus_engine";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr nexus_create_engine(ulong tier0Bytes, ulong tier1Bytes, ulong maxKvBlocks);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void nexus_destroy_engine(IntPtr engine);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void nexus_get_platform_info(ref NexusPlatformInfoRaw outInfo);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nexus_load_model_auto(IntPtr engine, [MarshalAs(UnmanagedType.LPStr)] string filepath);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nexus_prefetch_layer(IntPtr engine, [MarshalAs(UnmanagedType.LPStr)] string layerName);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nexus_deactivate_layer(IntPtr engine, [MarshalAs(UnmanagedType.LPStr)] string layerName);

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

        public static PlatformCapabilities GetPlatformCapabilities()
        {
            NexusPlatformInfoRaw raw = new NexusPlatformInfoRaw();
            nexus_get_platform_info(ref raw);
            return new PlatformCapabilities
            {
                OsName = raw.OsName,
                CpuArch = raw.CpuArch,
                VectorIsaName = raw.VectorIsaName,
                IsUnifiedMemory = raw.IsUnifiedMemory != 0,
                HasTransparentHugepages = raw.HasTransparentHugepages != 0,
                HardwareConcurrency = raw.HardwareConcurrency,
                TheoreticalBandwidthGbps = raw.TheoreticalBandwidthGbps,
                PerformanceTierDesc = raw.PerformanceTierDesc
            };
        }

        public bool LoadModel(string modelPath)
        {
            return nexus_load_model_auto(_handle, modelPath) == 0;
        }

        public bool PrefetchLayer(string layerName)
        {
            return nexus_prefetch_layer(_handle, layerName) == 0;
        }

        public bool DeactivateLayer(string layerName)
        {
            return nexus_deactivate_layer(_handle, layerName) == 0;
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
        }
    }
}
