use LesMills_Reporting
go 

declare @year as int 
set @year=2025;

WITH mw AS (
  SELECT
    m.MembershipID,
    CAST(m.WeekBeginningDate AS date) AS week,
    COALESCE(m.WeekVisits, 0)        AS engagement,
    CAST(m.WeekBeginningDate AS date)                         AS week_start,
    DATEADD(DAY, 6, CAST(m.WeekBeginningDate AS date))        AS week_end
  FROM pbi.LMNZ_MemberWeeklyAttendanceCount AS m
  inner join repo.CRM_ActiveMemberships crm 
  on m.MembershipID= crm.MembershipID
  where year(MembershipStartDate)>=@year and crm.MembershipStatusReason in ('Active', 'Active - Suspended')
),
susp AS (
  SELECT
    s.MembershipID,
    CAST(s.SuspensionStartDate AS date)                             AS pause_start,
    CAST(COALESCE(s.SuspensionEndDate, '9999-12-31') AS date)       AS pause_end
  FROM repo.MSReport_List_Suspensions AS s
)
SELECT 
   mw.MembershipID  AS member_id,
  mw.week,
  mw.engagement,
  CASE WHEN EXISTS (
         SELECT 1
         FROM susp s
         WHERE s.MembershipID = mw.MembershipID
           AND s.pause_start <= mw.week_end   -- pause starts before week ends
           AND s.pause_end   >= mw.week_start -- pause ends after week starts
       )
       THEN 1 ELSE 0 END AS paused
FROM mw 
ORDER BY member_id, week;
