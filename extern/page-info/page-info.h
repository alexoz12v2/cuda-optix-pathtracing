#if defined(DMT_OS_LINUX)
/*
 * page-info.h
 */

#ifndef PAGE_INFO_H_
#define PAGE_INFO_H_

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        /* page frame number: if present, the physical frame for the page */
        uint64_t pfn;
        /* soft-dirty set */
        bool softdirty;
        /* exclusively mapped, see e.g., https://patchwork.kernel.org/patch/6787921/ */
        bool exclusive;
        /* is a file mapping */
        bool file;
        /* page is swapped out */
        bool swapped;
        /* page is present, i.e, a physical page is allocated */
        bool present;
        /* if true, the kpageflags were successfully loaded, if false they were not (and are all zero) */
        bool kpageflags_ok;
        /* the 64-bit flag value extracted from /proc/kpageflags only if pfn is non-null */
        uint64_t kpageflags;

    } page_info;
    /*
 * Information for a number of virtually consecutive pages.
 */
    typedef struct
    {
        /* how many page_info structures are in the array pointed to by info */
        size_t num_pages;

        /* pointer to the array of page_info structures */
        page_info* info;
    } page_info_array;


    typedef struct
    {
        /* the number of pages on which this flag was set, always <= pages_available */
        size_t pages_set;

        /* the number of pages on which information could be obtained */
        size_t pages_available;

        /* the total number of pages examined, which may be greater than pages_available if
     * the flag value could not be obtained for some pages (usually because the pfn is not available
     * since the page is not yet present or because running as non-root.
     */
        size_t pages_total;

        /* the flag the values were queried for */
        int flag;

    } flag_count;

    /**
 * Examine the page info in infos to count the number of times a specified /proc/kpageflags flag was set,
 * effectively giving you a ratio, so you can say "80% of the pages for this allocation are backed by
 * huge pages" or whatever.
 *
 * The flags *must* come from kpageflags (these are not the same as those in /proc/pid/pagemap) and
 * are declared in linux/kernel-page-flags.h.
 *
 * Ideally, the flag information is available for all the pages in the range, so you can
 * say something about the entire range, but this is often not the case because (a) flags
 * are not available for pages that aren't present and (b) flags are generally never available
 * for non-root users. So the ratio structure indicates both the total number of pages as
 * well as the number of pages for which the flag information was available.
 */
    flag_count get_flag_count(page_info_array infos, int flag);

    /**
 * Given the case-insensitive name of a flag, return the flag number (the index of the bit
 * representing this flag), or -1 if the flag is not found. The "names" of the flags are
 * the same as the macro names in <linux/kernel-page-flags.h> without the KPF_ prefix.
 *
 * For example, the name of the transparent hugepages flag is "THP" and the corresponding
 * macro is KPF_THP, and the value of this macro and returned by this method is 22.
 *
 * You can generate the corresponding mask value to check the flag using (1ULL << value).
 */
    int flag_from_name(char const* name);

    /**
 * Print the info in the page_info structure to stdout.
 */
    void print_info(page_info info);

    /**
 * Print the info in the page_info structure to the give file.
 */
    void fprint_info(FILE* file, page_info info);


    /**
 * Print the table header that lines up with the tabluar format used by the "table" printing
 * functions. Called by fprint_ratios, or you can call it yourself if you want to prefix the
 * output with your own columns.
 */
    void fprint_info_header(FILE* file);

    /* print one info in a tabular format (as a single row) */
    void fprint_info_row(FILE* file, page_info info);


    /**
 * Print the ratio for each flag in infos. The ratio is the number of times the flag was set over
 * the total number of pages (or the total number of pages for which the information could be obtained).
 */
    void fprint_ratios_noheader(FILE* file, page_info_array infos);
    /*
 * Print a table with one row per page from the given infos.
 */
    void fprint_ratios(FILE* file, page_info_array infos);

    /*
 * Prints a summary of all the pages in the given array as ratios: the fraction of the time the given
 * flag was set.
 */
    void fprint_table(FILE* f, page_info_array infos);


    /**
 * Get info for a single page indicated by the given pointer (which may point anywhere in the page).
 */
    page_info get_page_info(void* p);

    /**
 * Get information for each page in the range from start (inclusive) to end (exclusive).
 */
    page_info_array get_info_for_range(void* start, void* end);

    /**
 * Free the memory associated with the given page_info_array. You shouldn't use it after this call.
 */
    void free_info_array(page_info_array infos);

#ifdef __cplusplus
}
#endif

#if defined(PAGE_INFO_IMPL)
/*
 * smaps.c
 *
 *  Created on: Jan 31, 2017
 *      Author: tdowns
 */

#include "page-info.h"

#include <assert.h>
#include <err.h>
#include <limits.h>
#include <linux/kernel-page-flags.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include <unistd.h>


#define PM_PFRAME_MASK    ((1ULL << 55) - 1)
#define PM_SOFT_DIRTY     (1ULL << 55)
#define PM_MMAP_EXCLUSIVE (1ULL << 56)
#define PM_FILE           (1ULL << 61)
#define PM_SWAP           (1ULL << 62)
#define PM_PRESENT        (1ULL << 63)

#ifdef __cplusplus
extern "C"
{
#endif

    /** bundles a flag with its description */
    typedef struct
    {
        int         flag_num;
        char const* name;
        bool        show_default;
    } flag;

#define FLAG_SHOW(name) {KPF_##name, #name, true},
#define FLAG_HIDE(name) {KPF_##name, #name, false},

    flag const kpageflag_defs[] = {
        FLAG_SHOW(LOCKED) FLAG_HIDE(ERROR) FLAG_HIDE(REFERENCED) FLAG_HIDE(UPTODATE) FLAG_HIDE(DIRTY) FLAG_HIDE(
            LRU) FLAG_SHOW(ACTIVE) FLAG_SHOW(SLAB) FLAG_HIDE(WRITEBACK) FLAG_HIDE(RECLAIM) FLAG_SHOW(BUDDY) FLAG_SHOW(MMAP)
            FLAG_SHOW(ANON) FLAG_SHOW(SWAPCACHE) FLAG_SHOW(SWAPBACKED) FLAG_SHOW(COMPOUND_HEAD) FLAG_SHOW(COMPOUND_TAIL)
                FLAG_SHOW(HUGE) FLAG_SHOW(UNEVICTABLE) FLAG_SHOW(HWPOISON) FLAG_SHOW(NOPAGE) FLAG_SHOW(KSM) FLAG_SHOW(THP)
/* older kernels won't have these new flags, so conditionally compile in support for them */
#ifdef KPF_BALLOON
                    FLAG_SHOW(BALLOON)
#endif
#ifdef KPF_ZERO_PAGE
                        FLAG_SHOW(ZERO_PAGE)
#endif
#ifdef KPF_IDLE
                            FLAG_SHOW(IDLE)
#endif

                                {-1, 0, false} // sentinel
    };

#define kpageflag_count (sizeof(kpageflag_defs) / sizeof(kpageflag_defs[0]) - 1)

#define ITERATE_FLAGS for (flag const* f = kpageflag_defs; f->flag_num != -1; f++)


// x-macro for doing some operation on all the pagemap flags
#define PAGEMAP_X(fn) fn(softdirty) fn(exclusive) fn(file) fn(swapped) fn(present)

    static unsigned get_page_size()
    {
        long psize = sysconf(_SC_PAGESIZE);
        assert(psize >= 1 && psize <= UINT_MAX);
        return (unsigned)psize;
    }

    /* round the given pointer down to the page boundary (i.e,. return a pointer to the page it lives in) */
    static inline void* pagedown(void* p, unsigned psize)
    {
        return (void*)(((uintptr_t)p) & -(uintptr_t)psize);
    }

    /**
 * Extract the interesting info from a 64-bit pagemap value, and return it as a page_info.
 */
    page_info extract_info(uint64_t bits)
    {
        page_info ret = {};
        ret.pfn       = bits & PM_PFRAME_MASK;
        ret.softdirty = bits & PM_SOFT_DIRTY;
        ret.exclusive = bits & PM_MMAP_EXCLUSIVE;
        ret.file      = bits & PM_FILE;
        ret.swapped   = bits & PM_SWAP;
        ret.present   = bits & PM_PRESENT;
        return ret;
    }

    /* print page_info to the given file */
    void fprint_info(FILE* f, page_info info)
    {
        fprintf(f,
                "PFN: %p\n"
                "softdirty = %d\n"
                "exclusive = %d\n"
                "file      = %d\n"
                "swapped   = %d\n"
                "present   = %d\n",
                (void*)info.pfn,
                info.softdirty,
                info.exclusive,
                info.file,
                info.swapped,
                info.present);
    }

    void print_info(page_info info)
    {
        fprint_info(stdout, info);
    }

    flag_count get_flag_count(page_info_array infos, int flag_num)
    {
        flag_count ret = {};

        if (flag_num < 0 || flag_num > 63)
        {
            return ret;
        }

        uint64_t flag = (1ULL << flag_num);

        ret.flag        = flag_num;
        ret.pages_total = infos.num_pages;

        for (size_t i = 0; i < infos.num_pages; i++)
        {
            page_info info = infos.info[i];
            if (info.kpageflags_ok)
            {
                ret.pages_set += (info.kpageflags & flag) == flag;
                ret.pages_available++;
            }
        }
        return ret;
    }

    /**
 * Print the table header that lines up with the tabluar format used by the "table" printing
 * functions. Called by fprint_ratios, or you can call it yourself if you want to prefix the
 * output with your own columns.
 */
    void fprint_info_header(FILE* file)
    {
        fprintf(file, "         PFN  sdirty   excl   file swappd presnt ");
        ITERATE_FLAGS
        {
            if (f->show_default)
                fprintf(file, "%4.4s ", f->name);
        }
        fprintf(file, "\n");
    }

    /* print one info in a tabular format (as a single row) */
    void fprint_info_row(FILE* file, page_info info)
    {
        fprintf(file, "%12p %7d%7d%7d%7d%7d ", (void*)info.pfn, info.softdirty, info.exclusive, info.file, info.swapped, info.present);

        if (info.kpageflags_ok)
        {
            ITERATE_FLAGS
            {
                if (f->show_default)
                    fprintf(file, "%4d ", !!(info.kpageflags & (1ULL << f->flag_num)));
            }
        }
        fprintf(file, "\n");
    }

#define DECLARE_ACCUM(name) size_t name##_accum = 0;
#define INCR_ACCUM(name)    name##_accum += info->name;
#define PRINT_ACCUM(name)   fprintf(file, "%7.4f", (double)name##_accum / infos.num_pages);


    void fprint_ratios_noheader(FILE* file, page_info_array infos)
    {
        PAGEMAP_X(DECLARE_ACCUM);
        size_t total_kpage_ok               = 0;
        size_t flag_totals[kpageflag_count] = {};
        for (size_t p = 0; p < infos.num_pages; p++)
        {
            page_info* info = &infos.info[p];
            PAGEMAP_X(INCR_ACCUM);
            if (info->kpageflags_ok)
            {
                total_kpage_ok++;
                int i = 0;
                ITERATE_FLAGS
                {
                    flag_totals[i++] += !!(info->kpageflags & (1ULL << f->flag_num));
                }
            }
        }

        printf("%12s ", "----------");
        PAGEMAP_X(PRINT_ACCUM)

        int i = 0;
        if (total_kpage_ok > 0)
        {
            ITERATE_FLAGS
            {
                if (f->show_default)
                    fprintf(file, " %4.2f", (double)flag_totals[i] / total_kpage_ok);
                i++;
            }
        }
        fprintf(file, "\n");
    }

    /*
 * Print a table with one row per page from the given infos.
 */
    void fprint_ratios(FILE* file, page_info_array infos)
    {
        fprint_info_header(file);
        fprint_ratios_noheader(file, infos);
    }

    /*
 * Prints a summary of all the pages in the given array as ratios: the fraction of the time the given
 * flag was set.
 */
    void fprint_table(FILE* f, page_info_array infos)
    {
        fprintf(f, "%zu total pages\n", infos.num_pages);
        fprint_info_header(f);
        for (size_t p = 0; p < infos.num_pages; p++)
        {
            fprint_info_row(f, infos.info[p]);
        }
    }


    /**
 * Get info for a single page indicated by the given pointer (which may point anywhere in the page)
 */
    page_info get_page_info(void* p)
    {
        // just get the info array for a single page
        page_info_array onepage = get_info_for_range(p, (char*)p + 1);
        assert(onepage.num_pages == 1);
        page_info ret = onepage.info[0];
        free_info_array(onepage);
        return ret;
    }

    /**
 * Get information for each page in the range from start (inclusive) to end (exclusive).
 */
    page_info_array get_info_for_range(void* start, void* end)
    {
        unsigned psize      = get_page_size();
        char*    start_page = (char*)pagedown(start, psize);
        char*    end_page   = (char*)pagedown((char*)end - 1, psize) + psize;
        size_t   page_count = start < end ? (end_page - start_page) / psize : 0;
        assert(page_count == 0 || start_page < end_page);

        if (page_count == 0)
        {
            return (page_info_array){0, NULL};
        }

        page_info* infos = (page_info*)malloc(page_count * sizeof(page_info));

        // open the pagemap file
        FILE* pagemap_file = fopen("/proc/self/pagemap", "rb");
        if (!pagemap_file)
            err(EXIT_FAILURE, "failed to open pagemap");

        // seek to the first page
        if (fseek(pagemap_file, (uintptr_t)start_page / psize * sizeof(uint64_t), SEEK_SET))
            err(EXIT_FAILURE, "pagemap seek failed");

        size_t    bitmap_bytes = page_count * sizeof(uint64_t);
        uint64_t* bitmap       = (uint64_t*)malloc(bitmap_bytes);
        assert(bitmap);
        size_t readc = fread(bitmap, bitmap_bytes, 1, pagemap_file);
        if (readc != 1)
            err(EXIT_FAILURE, "unexpected fread(pagemap) return: %zu", readc);

        fclose(pagemap_file);

        FILE* kpageflags_file = NULL;
        enum
        {
            INIT,
            OPEN,
            FAILED
        } file_state = INIT;

        for (size_t page_idx = 0; page_idx < page_count; page_idx++)
        {
            page_info info = extract_info(bitmap[page_idx]);

            if (info.pfn)
            {
                // we got a pfn, try to read /proc/kpageflags

                // open file if not open
                if (file_state == INIT)
                {
                    kpageflags_file = fopen("/proc/kpageflags", "rb");
                    if (!kpageflags_file)
                    {
                        warn("failed to open kpageflags");
                        file_state = FAILED;
                    }
                    else
                    {
                        file_state = OPEN;
                    }
                }

                if (file_state == OPEN)
                {
                    uint64_t bits;
                    if (fseek(kpageflags_file, info.pfn * sizeof(bits), SEEK_SET))
                        err(EXIT_FAILURE, "kpageflags seek failed");
                    if ((readc = fread(&bits, sizeof(bits), 1, kpageflags_file)) != 1)
                        err(EXIT_FAILURE, "unexpected fread(kpageflags) return: %zu", readc);
                    info.kpageflags_ok = true;
                    info.kpageflags    = bits;
                }
            }

            infos[page_idx] = info;
        }

        if (kpageflags_file)
            fclose(kpageflags_file);

        free(bitmap);

        return (page_info_array){page_count, infos};
    }

    void free_info_array(page_info_array infos)
    {
        free(infos.info);
    }

    int flag_from_name(char const* name)
    {
        ITERATE_FLAGS
        {
            if (strcasecmp(f->name, name) == 0)
            {
                return f->flag_num;
            }
        }
        return -1;
    }

#ifdef __cplusplus
}
#endif

#endif

#endif /* PAGE_INFO_H_ */
#endif